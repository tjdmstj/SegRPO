# GRPO reward #5 -- decoded MER from the CS-fine-tuned Whisper.
#
# This is the BASELINE the phoneme reward is meant to beat: score a rollout by actually
# transcribing it and measuring the mix error rate, the metric the paper reports.
#
# reward_cs_wer.py (#1) already tried "transcribe and compare" and had to crop every
# language span and decode it under its own tag, because stock ASR writes Korean-embedded
# English in Hangul -- so a full-utterance WER carried no gradient for the embedded
# language at all. That workaround is no longer needed. whisper_ft/ fine-tunes large-v3 on
# 20.5h of SwitchLingua and the failure is gone:
#
#     SwitchLingua test, real human recordings, stock -> fine-tuned
#       MER          28.74%  ->   5.87%
#       English lost  28.1%  ->    0.2%
#
# So this reward decodes the WHOLE utterance once, with the matrix-language token forced,
# and splits the errors by reference-token script afterwards using the aligner's own
# chunks. No cropping, no per-span ASR calls, no minimum span length.
#
# Two things to know before trusting it, both real and both reasons the phoneme reward
# exists:
#
#   1. LEAKAGE. The ASR was fine-tuned on data/<lang>/train, and run_grpo_kl.sh draws its
#      GRPO prompts from the SAME utterances. The scorer has therefore memorised the
#      reference text of every SwitchLingua prompt, which makes it lenient exactly where
#      the policy is being trained and can flatten the within-group spread. Point the RL
#      at cs-fleurs only (langs="" in the run script) for a clean comparison.
#   2. HACKABILITY. A decoded WER rewards slow over-articulation, and that is a failure
#      mode this corpus already shows -- the kor-eng SFT model's speaking rate fell 45%
#      while staying readable. Watch r_dur and the rollout durations.
#
# Offline check:
#   python reward_cs_mer.py --lst <testset>/Korean.lst --wav_dir <results>/<model> --limit 20
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torchaudio

from cs_reward_common import SpeakerSim, combine, duration_reward, mix_all
from reward_cs_whisper_ll import CAMPPLUS, WHISPER_LANG, parse_lang, read_lst

# the reward and the evaluation must score identically, so borrow the metric module the
# fine-tune ships rather than re-deriving the punctuation table and the zhconv rules
sys.path.insert(0, '')
from cs_metrics import CJK_CHAR, normalise, tokenize_cs   # noqa: E402

ADAPTER = ''


def bucket_of(token):
    """A reference token belongs to the matrix language or to the embedded English."""
    if not token:
        return None
    if CJK_CHAR.match(token[0]):
        return 'matrix'
    if any(c.isalpha() and c.isascii() for c in token):
        return 'en'
    return None                                    # digits, stray symbols: neither


def split_errors(ref_str, hyp_str, ins_weight=0.5):
    """-> {bucket: (weighted_errors, n_ref_tokens)} plus the overall MER.

    Errors are attributed through the aligner's own chunks, so a substitution is charged
    to the language of the reference token it replaced. Insertions have no reference token;
    they are charged to the bucket of the token they follow, which is where a hallucinated
    word actually sits.
    """
    from jiwer import process_words
    ref_t, hyp_t = ref_str.split(' '), hyp_str.split(' ')
    buckets = [bucket_of(t) for t in ref_t]
    err = {'en': 0.0, 'matrix': 0.0}
    n = {'en': 0, 'matrix': 0}
    for b in buckets:
        if b:
            n[b] += 1
    out = process_words(ref_str, hyp_str)
    for chunk in out.alignments[0]:
        if chunk.type == 'equal':
            continue
        if chunk.type == 'insert':
            i = min(max(chunk.ref_start_idx - 1, 0), len(buckets) - 1) if buckets else None
            b = buckets[i] if i is not None else None
            if b:
                err[b] += ins_weight * (chunk.hyp_end_idx - chunk.hyp_start_idx)
        else:                                       # substitute / delete
            for i in range(chunk.ref_start_idx, min(chunk.ref_end_idx, len(buckets))):
                if buckets[i]:
                    err[buckets[i]] += 1.0
    return err, n, float(out.wer)


class CSMerReward:
    """Drop-in sibling of CSWhisperLLReward / CSPhonemeReward / CSWerReward.

    __call__(wav, sr, gt_text, lang, ref_wav, ref_sr) -> dict(reward, r_content, r_matrix,
    r_embedded, ...), so cosyvoice/utils/grpo_executor.py cannot tell them apart.
    """

    def __init__(self, model='openai/whisper-large-v3', adapter=ADAPTER, device='cuda:0',
                 dtype='float16', alpha=1.0, beta=2.0, ins_weight=0.5, sharpness=3.0,
                 w_content=1.0, w_spk=0.0, w_dur=0.0, campplus_path=CAMPPLUS,
                 num_beams=1, max_new_tokens=200, content_scope='split'):
        from transformers import (WhisperFeatureExtractor,
                                  WhisperForConditionalGeneration, WhisperTokenizerFast)
        self.device = device
        td = dict(float16=torch.float16, bfloat16=torch.bfloat16,
                  float32=torch.float32)[dtype]
        if not str(device).startswith('cuda'):
            td = torch.float32
        self.fe = WhisperFeatureExtractor.from_pretrained(model)
        self.tok = WhisperTokenizerFast.from_pretrained(model)
        m = WhisperForConditionalGeneration.from_pretrained(model, torch_dtype=td)
        if adapter:
            # peft 0.20 raises on this box's torchao 0.9 instead of returning
            # False, so get_peft_model/from_pretrained dies before a single LoRA
            # layer is built. whisper_ft/peft_compat.py neutralises the probe.
            import peft_compat  # noqa: F401  (must precede peft)
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, adapter)
            m = m.merge_and_unload()      # fold LoRA in: generate() then runs at full speed
        m.config.forced_decoder_ids = None
        m.generation_config.forced_decoder_ids = None
        self.model = m.to(device).eval()
        self.dt = td
        if content_scope not in ('split', 'all'):
            raise ValueError(f'content_scope must be split or all: {content_scope}')
        # 'split'  combine(r_matrix, r_embedded, alpha, beta) -- the two languages scored
        #          apart and remixed, English up-weighted by beta:alpha
        # 'all'    1 - tanh(sharpness * MER) over the whole utterance, i.e. the number
        #          seed-tts-eval/cal_mer_whisper_ft.sh reports, verbatim
        #
        # The split was originally a WORKAROUND, not a modelling choice: before the scorer
        # was fine-tuned it lost 28.1% of English on real human speech, so reward_cs_wer.py
        # had to crop each span and transcribe it under its own language tag. LoRA took
        # that to 0.2% and one full-utterance decode now carries both languages, which is
        # what makes 'all' a usable option at all.
        #
        # r_matrix and r_embedded are returned either way. Under 'all' they are OBSERVED
        # ONLY -- they still get TRAIN/r_* and TRAIN/reward_std_* curves, but the objective
        # is the undivided MER.
        self.content_scope = content_scope
        self.alpha, self.beta = alpha, beta
        self.ins_weight, self.sharpness = ins_weight, sharpness
        self.w_content, self.w_spk, self.w_dur = w_content, w_spk, w_dur
        self.num_beams, self.max_new_tokens = num_beams, max_new_tokens
        self.spk = SpeakerSim(campplus_path, device=device) if w_spk > 0 else None
        self._prefix = {}

    def _to16k(self, wav, sr):
        if isinstance(wav, str):
            wav, sr = torchaudio.load(wav)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        return wav.squeeze(0)

    def _forced(self, code):
        if code not in self._prefix:
            ids = self.tok.convert_tokens_to_ids(
                [f'<|{code}|>', '<|transcribe|>', '<|notimestamps|>'])
            self._prefix[code] = [(i + 1, t) for i, t in enumerate(ids)]
        return self._prefix[code]

    @torch.inference_mode()
    def transcribe(self, wavs16k, code):
        feats = self.fe([w.cpu().numpy() for w in wavs16k], sampling_rate=16000,
                        return_tensors='pt').input_features
        ids = self.model.generate(feats.to(self.device, self.dt),
                                  forced_decoder_ids=self._forced(code),
                                  num_beams=self.num_beams,
                                  max_new_tokens=self.max_new_tokens)
        return self.tok.batch_decode(ids, skip_special_tokens=True)

    def _r(self, err, n):
        if not n:
            return None
        # 1 - tanh(sharpness * err_rate), the convention reward_cs_wer.py and
        # reward_tts.py already use, so the three rewards live on one scale
        return float(max(0.0, 1.0 - np.tanh(self.sharpness * err / n)))

    def __call__(self, wav, sr, gt_text, lang=None, ref_wav=None, ref_sr=None):
        code = WHISPER_LANG.get(parse_lang(lang), 'en') if lang else 'en'
        g = self._to16k(wav, sr)
        hyp = self.transcribe([g], code)[0]
        ref_s = normalise(gt_text, code)
        hyp_s = normalise(hyp, code)
        if not ref_s.strip():
            return {'reward': 0.0, 'r_content': 0.0, 'r_matrix': None,
                    'r_embedded': None, 'r_all': None, 'r_spk': None, 'r_dur': None,
                    'mer': None, 'hyp': hyp}
        err, n, mer = split_errors(ref_s, hyp_s, self.ins_weight)
        r_m, r_e = self._r(err['matrix'], n['matrix']), self._r(err['en'], n['en'])
        r_all = float(max(0.0, 1.0 - np.tanh(self.sharpness * mer)))
        r_content = r_all if self.content_scope == 'all' else combine(
            r_m, r_e, self.alpha, self.beta)
        r_spk = r_dur = None
        if ref_wav is not None:
            r = self._to16k(ref_wav, ref_sr or 16000)
            r_dur = duration_reward(g.numel() / 16000, r.numel() / 16000)
            if self.spk is not None:
                r_spk = self.spk(g.unsqueeze(0), r.unsqueeze(0))
        return {'reward': mix_all(r_content, r_spk, r_dur,
                                  self.w_content, self.w_spk, self.w_dur),
                'r_content': r_content, 'r_matrix': r_m, 'r_embedded': r_e,
                'r_all': r_all,
                'r_spk': r_spk, 'r_dur': r_dur,
                'mer': mer, 'n_matrix': n['matrix'], 'n_en': n['en'], 'hyp': hyp}


def main():
    ap = argparse.ArgumentParser(description='Offline check of the decoded-MER reward')
    ap.add_argument('--lst', required=True)
    ap.add_argument('--wav_dir', default=None,
                    help='default: score the lst audio itself (ground-truth topline)')
    ap.add_argument('--lang', default=None, help='default: per utterance, from the utt id')
    ap.add_argument('--model', default='openai/whisper-large-v3')
    ap.add_argument('--adapter', default=ADAPTER)
    ap.add_argument('--no_adapter', action='store_true', help='stock whisper, for contrast')
    ap.add_argument('--limit', type=int, default=20)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    rw = CSMerReward(model=args.model, adapter=None if args.no_adapter else args.adapter,
                     device=args.device)
    rows = read_lst(args.lst)[:args.limit]
    R, RM, RE, M = [], [], [], []
    for r in rows:
        p = (os.path.join(args.wav_dir, r['utt_id'] + '.wav') if args.wav_dir
             else (r['gt_wav'] or r['prompt_wav']))
        if not os.path.exists(p):
            continue
        wav, sr = torchaudio.load(p)
        out = rw(wav, sr, r['text'], args.lang or r['utt_id'])
        R.append(out['r_content'])
        M.append(out['mer'])
        if out['r_matrix'] is not None:
            RM.append(out['r_matrix'])
        if out['r_embedded'] is not None:
            RE.append(out['r_embedded'])
        f = lambda v: '-' if v is None else format(v, '.3f')
        print(f"[{r['utt_id']}] R={out['r_content']:.3f} matrix={f(out['r_matrix'])} "
              f"embedded={f(out['r_embedded'])} MER={out['mer'] * 100:.1f}% "
              f"n_en={out['n_en']}")
        if args.verbose:
            print(f"    REF: {r['text'][:110]}")
            print(f"    HYP: {out['hyp'][:110]}")
    if R:
        m = lambda v: float('nan') if not v else float(np.mean(v))
        print(f"\nmean R {m(R):.3f} | matrix {m(RM):.3f} | embedded {m(RE):.3f} "
              f"| MER {m(M) * 100:.2f}% | n={len(R)}")


if __name__ == '__main__':
    main()
