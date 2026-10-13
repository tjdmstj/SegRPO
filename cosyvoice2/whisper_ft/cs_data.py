"""SwitchLingua kaldi dirs -> Whisper training tensors.

The corpus already ships the three splits, disjoint by recording session:

    train  10,854 utts  20.45h   2,402 sessions
    dev       240 utts   0.44h      47 sessions
    test      709 utts   1.42h     162 sessions

`test` is the same audio the TTS evaluation set is drawn from (all 547 target utterances
and all 147 prompts of data/cs_testset_sft/meta.lst), and neither train nor dev touches
it. That is what makes this fine-tune usable as the TTS scorer: the ASR never sees the
utterances it will later be asked to transcribe.

Features are computed in the collator rather than cached with dataset.map(). large-v3
uses 128 mel bins, so caching 10,854 x 128 x 3000 float32 would be ~17 GB on disk to
save a transform that costs a few ms.
"""
from __future__ import annotations

import os

import torch
import torchaudio

from cs_metrics import LANG2WHISPER, WHISPER2NAME

SPLITS = ('train', 'dev', 'test')


def read_kv(path, join=False):
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split(' ')
            if len(parts) < 2:
                continue
            out[parts[0]] = ' '.join(parts[1:]) if join else parts[1]
    return out


def read_split(data_root, langs, split, utt_filter=None):
    """-> [{utt_id, wav, text, lang, whisper_lang}] over the requested languages."""
    rows = []
    for lang in langs:
        d = os.path.join(data_root, lang, split)
        if not os.path.isdir(d):
            continue
        wav = read_kv(os.path.join(d, 'wav.scp'))
        txt = read_kv(os.path.join(d, 'text'), join=True)
        for utt, path in sorted(wav.items()):
            if utt not in txt or not txt[utt].strip():
                continue
            if utt_filter is not None and utt not in utt_filter:
                continue
            rows.append({'utt_id': utt, 'wav': path, 'text': txt[utt].strip(),
                         'lang': lang, 'whisper_lang': LANG2WHISPER[lang]})
    return rows


def read_kaldi_dir(d, lang=None):
    """Any kaldi dir with wav.scp + text -> the same row format as read_split().

    Used for cs-fleurs, whose layout is data_csfleurs/<subset>/<pair>/test/. The language
    is taken per utterance from the id unless one is given explicitly.
    """
    from cs_metrics import lang_of_utt
    wav = read_kv(os.path.join(d, 'wav.scp'))
    txt = read_kv(os.path.join(d, 'text'), join=True)
    rows = []
    for utt, path in sorted(wav.items()):
        if utt not in txt or not txt[utt].strip():
            continue
        l = lang or lang_of_utt(utt)
        if l is None or l not in LANG2WHISPER:
            continue
        rows.append({'utt_id': utt, 'wav': path, 'text': txt[utt].strip(),
                     'lang': l, 'whisper_lang': LANG2WHISPER[l]})
    return rows


def load_audio_16k(path):
    wav, sr = torchaudio.load(path)
    if wav.dim() > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0).numpy()


class CSDataset(torch.utils.data.Dataset):
    """One item -> (log-mel features, label ids).

    The label prefix carries the utterance's OWN language token, built explicitly rather
    than through tokenizer.set_prefix_tokens(), which is stateful and would race across
    dataloader workers. The leading <|startoftranscript|> is dropped here because
    WhisperForConditionalGeneration prepends decoder_start_token_id itself.
    """

    def __init__(self, rows, feature_extractor, tokenizer, max_label_len=448):
        self.rows = rows
        self.fe = feature_extractor
        self.tok = tokenizer
        self.max_label_len = max_label_len
        self.prefix = {}
        for code in set(LANG2WHISPER.values()):
            ids = [tokenizer.convert_tokens_to_ids(f'<|{code}|>'),
                   tokenizer.convert_tokens_to_ids('<|transcribe|>'),
                   tokenizer.convert_tokens_to_ids('<|notimestamps|>')]
            assert tokenizer.unk_token_id not in ids, f'checkpoint has no <|{code}|>'
            self.prefix[code] = ids
        self.eot = tokenizer.convert_tokens_to_ids('<|endoftext|>')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        audio = load_audio_16k(r['wav'])
        feats = self.fe(audio, sampling_rate=16000).input_features[0]
        body = self.tok(r['text'], add_special_tokens=False).input_ids
        labels = self.prefix[r['whisper_lang']] + body + [self.eot]
        return {'input_features': feats, 'labels': labels[:self.max_label_len],
                'utt_id': r['utt_id']}


class Collator:
    """Pad features and labels; masked label positions become -100 so they do not train."""

    def __init__(self, feature_extractor, tokenizer):
        self.fe = feature_extractor
        self.tok = tokenizer

    def __call__(self, batch):
        feats = self.fe.pad([{'input_features': b['input_features']} for b in batch],
                            return_tensors='pt')
        labels = self.tok.pad([{'input_ids': b['labels']} for b in batch],
                              return_tensors='pt')
        lab = labels['input_ids'].masked_fill(labels.attention_mask.ne(1), -100)
        return {'input_features': feats['input_features'], 'labels': lab}


def language_name(code):
    return WHISPER2NAME[code]
