"""Phone recognizers behind one interface, for the audio-reference reward.

Both places that touch phones must use the SAME model, because the reference sequence and
the rollout score are only comparable in one inventory:

    build_audio_refs.py   original recording -> free decode -> refs/*.pt
    reward_cs_audioref.py rollout -> emission -> CTC against those refs

    load_phone_model(name, device, dtype) -> PhoneModel
        .logits(wav16k)  -> [T, V] raw frame logits, 20 ms per frame
        .vocab           {phone: id}
        .blank           CTC blank id
        .name            model id, stamped into every ref entry

Backends
--------
facebook/wav2vec2-lv-60-espeak-cv-ft   (default, what every existing refs/*.pt was built with)
    316M, 392 units in espeak's IPA. 85 units carry tone digits (i5, onɡ5, ...), so Mandarin
    and Cantonese tone is part of the target.

changelinglab/PhoneticXeus             (Bharadwaj et al., arXiv:2603.29042)
    575M, XEUS encoder + self-conditioned CTC, trained on IPAPack++ (70+ languages).
    428 units: <blank>=0, <sos>, <eos>, <unk>, then IPA phones with diacritics (ʰ ʲ ː ̤ ...).
    NO tone units at all -- tone is invisible to a reward built on it, which matters for
    Mandarin and Cantonese. Phonetic (what was said) rather than phonemic, which suits a
    target read off a real recording.
    The forward returns RAW logits; the waveform is normalised inside the model (x10
    amplitude moves the logits by <0.06), so no feature extractor is applied here.
    Needs trust_remote_code. First load downloads ~2.3 GB.
"""
from __future__ import annotations

import json
import os

import torch

WAV2VEC2_ESPEAK = 'facebook/wav2vec2-lv-60-espeak-cv-ft'
PHONETIC_XEUS = 'changelinglab/PhoneticXeus'


def _hub_file(model, fname):
    p = os.path.join(model, fname)
    if os.path.exists(p):
        return p
    from huggingface_hub import hf_hub_download
    return hf_hub_download(model, fname)


def is_phoneticxeus(name):
    return 'phoneticxeus' in str(name).lower()


class PhoneModel:
    def __init__(self, name=WAV2VEC2_ESPEAK, device='cuda:0', dtype='float32'):
        self.name = name
        self.device = device
        self.dt = dict(float16=torch.float16, float32=torch.float32)[dtype]
        if is_phoneticxeus(name):
            from transformers import AutoModel
            self.kind = 'phoneticxeus'
            self.fe = None
            # fp32 regardless of dtype: the remote code casts inputs to self.dtype and CTC
            # is summed in log space downstream, where fp16 underflows
            self.model = AutoModel.from_pretrained(
                name, trust_remote_code=True).to(device).eval()
            self.vocab = json.load(open(_hub_file(name, 'ipa_vocab.json'), encoding='utf-8'))
            self.blank = int(self.vocab.get('<blank>', 0))
        else:
            from transformers import AutoFeatureExtractor, AutoModelForCTC
            self.kind = 'wav2vec2'
            self.fe = AutoFeatureExtractor.from_pretrained(name)
            self.model = AutoModelForCTC.from_pretrained(
                name, torch_dtype=self.dt).to(device).eval()
            self.vocab = json.load(open(_hub_file(name, 'vocab.json'), encoding='utf-8'))
            self.blank = self.model.config.pad_token_id or 0
        self.id2p = {v: k for k, v in self.vocab.items()}
        self.n_vocab = len(self.vocab) if self.kind == 'phoneticxeus' \
            else self.model.config.vocab_size

    @torch.inference_mode()
    def logits(self, wav16k):
        """wav16k: 1-D float tensor or array at 16 kHz -> [T, V] float32 logits."""
        if not torch.is_tensor(wav16k):
            wav16k = torch.as_tensor(wav16k)
        wav16k = wav16k.float().reshape(-1)
        if self.kind == 'phoneticxeus':
            out = self.model(input_values=wav16k.unsqueeze(0).to(self.device)).logits
        else:
            x = self.fe(wav16k.cpu().numpy(), sampling_rate=16000,
                        return_tensors='pt').input_values
            out = self.model(x.to(self.device, self.dt)).logits
        return out[0].float()


def load_phone_model(name=WAV2VEC2_ESPEAK, device='cuda:0', dtype='float32'):
    return PhoneModel(name=name, device=device, dtype=dtype)
