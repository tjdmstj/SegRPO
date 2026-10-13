# Shared machinery for the two code-switching GRPO rewards.
#
# Both rewards need the same thing first: given the generated audio and the ground-truth
# text, find WHERE each language is on the time axis. A single-language-tag ASR cannot do
# this -- SenseVoice writes Korean-embedded English in Hangul, so its Latin preservation on
# kor-eng is 3% even on ground-truth audio, which makes an utterance-level WER blind to the
# embedded language. Forced alignment sidesteps that: the transcript is already known, so
# the model only has to place known units on the time axis.
#
# torchaudio's MMS_FA covers 1100+ languages with one model, so every language in the
# corpus goes through an identical code path.
#
# Known limitation: uroman romanizes Japanese kanji with Chinese readings
# (合金とは -> "hejintoha", not "gokin towa"), so Japanese MATRIX spans align poorly
# (observed CTC score 0.03 on 2種類以上の). Japanese ENGLISH spans are unaffected
# (0.62-0.89), so R_embedded is usable for Japanese but R_matrix is not. Install
# pykakasi and pre-convert kanji to kana if Japanese matrix scoring is needed.
from __future__ import annotations

import re
import string
import unicodedata

import numpy as np
import torch
import torchaudio

# matrix language -> SenseVoice tag. Every utterance in the corpus is code-switched with
# English, so the embedded tag is always 'en'.
LANG_TAG = {
    'Mandarin': 'zh', 'cmn-eng': 'zh', 'cmn': 'zh', 'zh': 'zh',
    'Cantonese': 'yue', 'yue-eng': 'yue', 'yue': 'yue',
    'Japanese': 'ja', 'jpn-eng': 'ja', 'jpn': 'ja', 'ja': 'ja',
    'Korean': 'ko', 'kor-eng': 'ko', 'kor': 'ko', 'ko': 'ko',
}

LATIN = re.compile(r"[A-Za-z][A-Za-z'\-]*")
PUNCT = set(string.punctuation) | set('。，、！？；：""''（）《》〈〉【】…—～·「」『』〜｡､！？：；')


def guess_lang(text):
    """Fall back to script detection when the caller does not pass a language."""
    for ch in text:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7AF:
            return 'ko'
        if 0x3040 <= o <= 0x30FF:
            return 'ja'
    for ch in text:
        if 0x4E00 <= ord(ch) <= 0x9FFF:
            return 'zh'  # cannot separate zh from yue by script; pass lang explicitly
    return 'en'


def split_spans(text):
    """-> [(surface, 'en'|'matrix')].

    Split by script rather than whitespace: code-switching happens inside a whitespace
    token too ('mixture이다' holds both languages).
    """
    spans, buf, cur = [], '', None
    i = 0
    while i < len(text):
        m = LATIN.match(text, i)
        if m:
            kind, piece = 'en', m.group(0)
            i = m.end()
        else:
            kind, piece = 'matrix', text[i]
            i += 1
        if piece.strip() == '':
            buf += piece
            continue
        if kind != cur and buf.strip():
            spans.append((buf, cur))
            buf = ''
        cur = kind
        buf += piece
    if buf.strip():
        spans.append((buf, cur))
    # Drop punctuation-only spans. A bare ',' between two English words becomes a ~0.02s
    # unit that the aligner scores 0.00, which silently drags R_matrix down.
    return [(s.strip(), k) for s, k in spans if s.strip() and _has_content(s)]


def _has_content(s):
    return any(c.isalnum() or is_cjk(c) for c in s)


def is_cjk(ch):
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or
            0x3040 <= o <= 0x309F or 0x30A0 <= o <= 0x30FF or
            0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF)


def normalize(text):
    text = unicodedata.normalize('NFKC', str(text))
    # SenseVoice's rich_transcription_postprocess turns emotion/event tags into emoji
    # ('Sligger.😔'), which would otherwise be counted as a token.
    text = ''.join(c for c in text if unicodedata.category(c) not in ('So', 'Sk', 'Cf'))
    text = ''.join(' ' if c in PUNCT else c for c in text)
    return re.sub(r'\s+', ' ', text).strip().lower()


def tokenize_mixed(text):
    """CJK per character, Latin per word -- the MER convention."""
    tokens, buf = [], []
    for ch in normalize(text):
        if is_cjk(ch):
            if buf:
                tokens.append(''.join(buf)); buf = []
            tokens.append(ch)
        elif ch.isspace():
            if buf:
                tokens.append(''.join(buf)); buf = []
        else:
            buf.append(ch)
    if buf:
        tokens.append(''.join(buf))
    return tokens


def edit_distance(ref, hyp):
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = prev[j - 1] if ref[i - 1] == hyp[j - 1] else 1 + min(prev[j - 1], prev[j], cur[j - 1])
        prev = cur
    return prev[m]


class SpanAligner:
    """Forced-align an utterance against its ground-truth text, one unit per language span.

    Returns [(surface, kind, start_sec, end_sec, ctc_score)]. The CTC score doubles as a
    pronunciation-canonicality signal: on ground truth, Korean-accented 'basically' scored
    0.26 while a clearly-articulated Mandarin-embedded 'Alloys' scored 0.81.
    """

    def __init__(self, device='cuda:0'):
        import uroman as ur
        self.uro = ur.Uroman()
        self.bundle = torchaudio.pipelines.MMS_FA
        self.model = self.bundle.get_model().to(device).eval()
        self.tokenizer = self.bundle.get_tokenizer()
        self.aligner = self.bundle.get_aligner()
        self.device = device
        self.sample_rate = self.bundle.sample_rate
        self.n_infeasible = 0   # rollouts too short to align; counted, not raised

    def _romanize(self, s):
        # MMS_FA's vocabulary is 29 Latin characters, so everything must be romanized.
        # '-' is MMS_FA's BLANK token (index 0). Letting it through makes the aligner
        # raise "targets Tensor shouldn't contain blank index". uroman emits it for
        # hyphenated words ('code-switching') and some scripts' syllable breaks.
        r = re.sub(r"[^a-z']", '', self.uro.romanize_string(s).lower())
        return r or 'a'   # the aligner rejects empty units

    def __call__(self, wav, sr, text):
        spans = split_spans(text)
        if not spans:
            return []
        roman = [self._romanize(s) for s, _ in spans]

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        with torch.inference_mode():
            emission, _ = self.model(wav.to(self.device))
            tokens = self.tokenizer(roman)
            # CTC cannot align a target longer than the emission (each repeated symbol also
            # needs a blank between). A GRPO rollout that stops early hits this constantly,
            # and it is a property of the rollout, not a bug: report "no spans" so the
            # caller scores it low instead of raising
            #   RuntimeError: targets length is too long for CTC
            flat = [t for w in tokens for t in w]
            need = len(flat) + sum(1 for a, b in zip(flat, flat[1:]) if a == b)
            if emission.size(1) < need:
                self.n_infeasible += 1
                return [], wav.squeeze(0)
            token_spans = self.aligner(emission[0], tokens)
        ratio = wav.size(1) / emission.size(1) / self.sample_rate

        out = []
        for (surface, kind), ts in zip(spans, token_spans):
            st, en = ts[0].start * ratio, ts[-1].end * ratio
            n = sum(len(t) for t in ts)
            score = sum(t.score * len(t) for t in ts) / n if n else 0.0
            out.append((surface, kind, float(st), float(en), float(score)))
        return out, wav.squeeze(0)


def combine(r_matrix, r_embedded, alpha=1.0, beta=2.0):
    """Weighted mix. beta > alpha because English is the minority of every utterance's
    duration yet is exactly the part that regressed after SFT (Hangul preservation
    93%->81%, English realised as katakana in Japanese)."""
    if r_matrix is None and r_embedded is None:
        return 0.0
    if r_embedded is None:
        return float(r_matrix)
    if r_matrix is None:
        return float(r_embedded)
    return float((alpha * r_matrix + beta * r_embedded) / (alpha + beta))


# --------------------------------------------------------------------------------------
# Audio-matching terms. These are independent of how the CONTENT reward is computed, so
# both reward_cs_wer.py and reward_cs_align.py mix in the same two.
#
# Without them the reward only asks "is this intelligible", never "does it sound like the
# target". SFT optimises against the ground-truth speech tokens, so it implicitly matches
# speaker and timing; a content-only RL objective on top of it can drift away from both.
# The failure we measured is exactly of that shape: on kor-eng the SFT model's speaking
# rate fell 45% and its F0 range grew 69% versus ground truth while staying "readable".
# --------------------------------------------------------------------------------------

class SpeakerSim:
    """Cosine similarity between campplus speaker embeddings of generated vs reference.

    Reuses the same campplus.onnx and the same fbank front-end as tools/extract_embedding.py,
    so the numbers are comparable to the utt2embedding.pt already on disk.
    """

    def __init__(self, onnx_path, device='cuda:0'):
        import onnxruntime
        opt = onnxruntime.SessionOptions()
        opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        opt.intra_op_num_threads = 1
        providers = ['CUDAExecutionProvider'] if str(device).startswith('cuda') else ['CPUExecutionProvider']
        self.sess = onnxruntime.InferenceSession(onnx_path, sess_options=opt, providers=providers)

    def embed(self, wav16k):
        import torchaudio.compliance.kaldi as kaldi
        if wav16k.dim() == 1:
            wav16k = wav16k.unsqueeze(0)
        feat = kaldi.fbank(wav16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        out = self.sess.run(None, {self.sess.get_inputs()[0].name:
                                   feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten()
        return out / (np.linalg.norm(out) + 1e-8)

    def __call__(self, gen_16k, ref_16k):
        a, b = self.embed(gen_16k), self.embed(ref_16k)
        cos = float(np.dot(a, b))
        # cosine lives in [-1, 1]; map to [0, 1] so it mixes with the other terms
        return max(0.0, min(1.0, (cos + 1.0) / 2.0))


def duration_reward(gen_sec, ref_sec, tol=0.20):
    """1.0 when the generated duration matches the reference, decaying with relative error.

    `tol` is the relative error that still scores ~0.5, so small natural variation is not
    punished while the 45% rate collapse we observed is.
    """
    if not ref_sec or ref_sec <= 0 or gen_sec <= 0:
        return None
    rel = abs(gen_sec - ref_sec) / ref_sec
    return float(max(0.0, 1.0 - rel / (2.0 * tol)))


def mix_all(r_content, r_spk, r_dur, w_content=1.0, w_spk=0.5, w_dur=0.3):
    """Renormalises over whichever terms are available, so a missing reference audio
    degrades gracefully to the content-only reward instead of dragging the total to 0."""
    num = den = 0.0
    for r, w in ((r_content, w_content), (r_spk, w_spk), (r_dur, w_dur)):
        if r is not None:
            num += w * r
            den += w
    return float(num / den) if den else 0.0
