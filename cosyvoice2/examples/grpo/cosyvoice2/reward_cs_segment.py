# GRPO reward -- per-SEGMENT character error rate, with each segment's place in the
# rollout's own audio.
#
# The whole-utterance reward gives one number to every token of a take, so a rollout that
# said the Korean well and the English badly is pushed the same way on both. reward_cs_mer_
# enspan.py narrowed that to the English, but still with ONE score for the whole English
# stretch: four takes that all get a word wrong -- seriez / serious / 시리즈 / seri -- score
# identically under word-level right/wrong, the group spread is zero and the word teaches
# nothing.
#
# Scoring each segment by CHARACTER error rate brings that word back to life:
#
#     series -> seriez    1/6   0.83
#     series -> serious   2/6   0.67
#     series -> seri      2/6   0.67
#     series -> 시리즈     saturates  0.00
#
# and the group now has a gradient from "nearly right" to "fully assimilated into Korean".
#
# Nothing here is restricted to English. Every segment is scored, and which segments end up
# carrying signal is decided by the DATA: the executor normalises each segment across the
# group, so a segment the four takes agree on has zero spread and contributes nothing, while
# a segment they disagree on does. English is where disagreement concentrates, but it is not
# hardcoded, and a matrix-language segment that the takes render differently is trained too.
#
# SEGMENT UNIT. MER's own token is one CJK/Hangul character or one Latin word. Used
# directly that is both too short to align (one Hangul character is 4-6 speech tokens, at
# the limit of what cross-attention resolves) and back to binary (a 1-character CER is 0 or
# 1, which is the degeneracy this reward exists to remove). So Latin words stay whole and
# CJK/Hangul runs are chunked, `seg_chunk_chars` at a time; seg_unit='mer' restores the
# metric's own unit exactly.
#
# TIMESTAMPS come from forced alignment of the REFERENCE onto the rollout, not from the
# decode's own word times. Every reference segment then has a span even when the take
# dropped it entirely, which is exactly the case that most needs a negative advantage, and
# it is the same Aligner whose span accuracy build_audio_refs.py already validated.
#
# Offline check:
#   python reward_cs_segment.py --lst <testset>/Korean.lst --wav_dir <results>/<model> --limit 3
from __future__ import annotations

import argparse
import logging
import os
import re
import sys

import numpy as np
import torch

from reward_cs_mer import CSMerReward
from reward_cs_mer_mos import Utmos22, UtmosV2, rescale
from reward_cs_whisper_ll import WHISPER_LANG, parse_lang, read_lst

WFT = ''
sys.path.insert(0, WFT)
from build_audio_refs import Aligner, dtw, median_filter  # noqa: E402
from cs_metrics import CJK_CHAR, normalise, tokenize_cs  # noqa: E402

TOKEN_HZ = 25.0


def is_cjk(tok):
    return bool(tok) and bool(CJK_CHAR.match(tok[0]))


# CosyVoice2's LLM tokenizer (Qwen2). SparkTTS passes its own policy's tokenizer instead.
BPE_TOKENIZER = ''


def ref_segments(norm_text, unit='chunk', chunk=3, tokenizer=None, raw_text=None):
    """Split the normalised reference into scoring segments.

    -> [(char_start, char_end, [token, ...], kind)] over tokenize_cs's tokens, which is the
    unit MER counts: one CJK/Hangul character, one Latin word.

    unit:
      'cs'      TWO GROUPS: everything in the embedded language against everything in the
                matrix language, however many times the sentence switches.
                '안녕하세요 meeting이 있습니다' scores [안녕하세요 + 이 있습니다] as one
                number and [meeting] as another. The spans stay separate (the aligner needs
                contiguous ranges), but they share a score and therefore an advantage, so
                every matrix token of the utterance moves together and every embedded token
                moves together. This is the coarsest segmentation that still separates the
                two languages, and it is the direct ablation of 'switch': same spans, same
                alignment, scores pooled per language instead of per run.
      'switch'  ONE SEGMENT PER LANGUAGE RUN. Everything between two code switches is a
                single segment, so the unit of the reward is the unit of the phenomenon:
                '안녕하세요 meeting이 있습니다' -> [안녕하세요][meeting][이 있습니다].
                Nothing is chosen by hand -- no chunk size to justify, and no segment that
                straddles a switch. Segments are long, so the alignment is easy and the
                character error rate inside one is finely graded; the cost is fewer
                segments, i.e. coarser localisation within a language run.
      'chunk'   Latin words whole, CJK/Hangul runs grouped `chunk` characters at a time.
                Measured on the dev alignments, an English word averages 0.399 s and a
                matrix character 0.271 s, so chunk=3 makes matrix segments 2.03x the
                length of English ones and chunk=2 makes them 1.35x -- the asymmetry this
                variant is trying to avoid.
      'mer'     the metric's own token: one CJK/Hangul character, one Latin word. Shortest
                and most faithful to MER, but a 1-character CER is binary and 0.271 s is at
                the limit of what the forced alignment resolves.
      'bpe'     'switch' first, then every language run cut again at the BPE token
                boundaries of the POLICY's own text tokenizer (`tokenizer`), so the unit of
                the reward is the unit the policy is conditioned on and nothing is chosen
                by hand. This includes Latin words: 'determinism' -> [determin][ism]. A
                piece of a word is scored on the characters of that word's recognition
                that a character-level alignment assigns to it (piece_hyps), and located in
                time inside the word by character proportion (segment_spans).
                `raw_text`, the un-normalised sentence, makes the English cuts the ones the
                policy actually sees: normalise() lower-cases, and 4.2% of the test-set
                English words split differently in lower case ('Hong' is one token,
                'hong' is 'h'+'ong').
    """
    if unit == 'bpe':
        if tokenizer is None:
            raise ValueError("seg_unit='bpe' needs the policy's text tokenizer")
        return _bpe_segments(norm_text, tokenizer, raw_text)
    if unit in ('switch', 'cs'):
        return _switch_segments(norm_text)
    segs, pos, run = [], 0, []
    for tok in tokenize_cs(norm_text):
        a = norm_text.find(tok, pos)
        if a < 0:
            continue
        pos = a + len(tok)
        if is_cjk(tok) and unit != 'mer':
            run.append((a, pos, tok))
            if len(run) >= chunk:
                segs.append((run[0][0], run[-1][1], [t for _, _, t in run], 'matrix'))
                run = []
        else:
            if run:
                segs.append((run[0][0], run[-1][1], [t for _, _, t in run], 'matrix'))
                run = []
            segs.append((a, pos, [tok], 'matrix' if is_cjk(tok) else 'en'))
    if run:
        segs.append((run[0][0], run[-1][1], [t for _, _, t in run], 'matrix'))
    return segs


def _switch_segments(norm_text):
    """One segment per maximal run of the same language, i.e. cut only where the language
    changes. Adjacent same-kind tokens are merged; the char range spans from the first to
    the last token of the run, so punctuation and spaces between them stay inside."""
    segs = []
    pos = 0
    for tok in tokenize_cs(norm_text):
        a = norm_text.find(tok, pos)
        if a < 0:
            continue
        pos = a + len(tok)
        kind = 'matrix' if is_cjk(tok) else 'en'
        if segs and segs[-1][3] == kind:
            s0, _, toks, k = segs[-1]
            segs[-1] = (s0, pos, toks + [tok], k)
        else:
            segs.append((a, pos, [tok], kind))
    return segs


def _word_cuts(word, tokenizer):
    """Internal BPE cut offsets of one Latin word, tokenised as ' word' (the usual case:
    English in these corpora is space-separated)."""
    e = tokenizer(' ' + word, add_special_tokens=False, return_offsets_mapping=True)
    return sorted({c - 1 for x, y in e['offset_mapping'] if y > x for c in (x, y)
                   if 1 < c < len(word) + 1})


def _bpe_segments(norm_text, tokenizer, raw_text=None):
    """Language runs (exactly _switch_segments), each cut again at BPE token boundaries.

    CJK/Hangul runs are re-joined as natural text before they are tokenised. normalise()
    puts a space between every CJK character, and fed that, a byte-level BPE splits each
    character into byte fragments and the boundaries mean nothing. Boundaries come from
    the tokenizer's character offsets; a character split into byte fragments has every
    fragment mapped onto it, so a cut can only fall BETWEEN characters.

    Latin words are cut where the tokenizer cuts them IN THE RAW SENTENCE: each normalised
    word is found case-insensitively in `raw_text` (in order) and the token boundaries that
    fall strictly inside it are its cuts. Where normalisation changed the word so that it
    cannot be found, the normalised word itself is tokenised instead.

    The run boundaries stay hard cuts, so a segment never mixes the two languages. (On the
    CS test sets every switch point already is a BPE boundary -- 13,368 of 13,368.)
    """
    raw_cuts, cursor = None, 0
    if raw_text:
        enc = tokenizer(raw_text, add_special_tokens=False, return_offsets_mapping=True)
        raw_cuts = {c for x, y in enc['offset_mapping'] if y > x for c in (x, y)}
    out = []
    for a, b, toks, kind in _switch_segments(norm_text):
        locs, pos = [], a
        for t in toks:
            i = norm_text.find(t, pos)
            locs.append((i, i + len(t)))
            pos = i + len(t)
        if kind == 'en':
            for (x, _), t in zip(locs, toks):
                cuts = None
                if raw_cuts is not None:
                    m = re.compile(re.escape(t), re.IGNORECASE).search(raw_text, cursor)
                    if m:
                        cursor = m.end()
                        cuts = sorted(c - m.start() for c in raw_cuts
                                      if m.start() < c < m.end())
                if cuts is None:
                    cuts = _word_cuts(t, tokenizer)
                edges = [0] + cuts + [len(t)]
                out.extend((x + p0, x + p1, [t[p0:p1]], kind)
                           for p0, p1 in zip(edges, edges[1:]) if p1 > p0)
            continue
        text, spans = '', []
        for t in toks:
            spans.append((len(text), len(text) + len(t)))
            text += t
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        cuts = {c for x, y in enc['offset_mapping'] if y > x for c in (x, y)}
        group = [0]
        for k in range(1, len(toks)):
            if spans[k - 1][1] in cuts:                # BPE boundary between k-1 and k
                out.append((locs[group[0]][0], locs[group[-1]][1],
                            [toks[i] for i in group], kind))
                group = [k]
            else:
                group.append(k)
        out.append((locs[group[0]][0], locs[group[-1]][1],
                    [toks[i] for i in group], kind))
    return out


def piece_hyps(ref_word, hyp, cuts):
    """Split the recognition of one reference word over the word's BPE pieces.

    A character-level alignment (Levenshtein opcodes) of `hyp` onto `ref_word`: a matched
    or substituted hypothesis character goes to the piece of the reference character it
    stands for (spread evenly when the two sides of a substitution differ in length), an
    inserted one to the piece on its left -- the same convention ref_to_hyp uses between
    tokens. A word recognised in another script ('meeting' -> '米听') spreads its
    characters over the pieces, and every piece then scores 0, as the whole word would.
    """
    from rapidfuzz.distance import Levenshtein
    edges = [0] + list(cuts) + [len(ref_word)]
    owner = [k for k in range(len(edges) - 1) for _ in range(edges[k], edges[k + 1])]
    out = [[] for _ in range(len(edges) - 1)]
    if not hyp:
        return [''] * len(out)
    for op in Levenshtein.opcodes(ref_word, hyp):
        i1, i2, j1, j2 = op.src_start, op.src_end, op.dest_start, op.dest_end
        if op.tag in ('equal', 'replace'):
            n_src, n_dst = i2 - i1, j2 - j1
            for k in range(n_dst):
                out[owner[i1 + (k * n_src) // n_dst]].append(hyp[j1 + k])
        elif op.tag == 'insert':
            out[owner[i1 - 1 if i1 > 0 else 0]].append(hyp[j1:j2])
    return [''.join(x) for x in out]


def segment_hyps(ref_n, ref_tok, mapped, segs):
    """-> per segment, the recognised tokens/characters that belong to it.

    A segment that covers whole reference tokens gets their aligned hypothesis tokens, as
    in every other unit. A segment that is a PIECE of a token (a BPE piece of a Latin
    word) gets its share of that token's recognition from piece_hyps.
    """
    locs, pos = [], 0
    for t in ref_tok:
        i = ref_n.find(t, pos)
        locs.append((i, i + len(t)))
        pos = i + len(t)
    pieces = {}                     # token index -> sorted internal cut offsets
    for a, b, _, _ in segs:
        for ti, (ta, tb) in enumerate(locs):
            if ta < b and tb > a and not (a <= ta and tb <= b):
                cs = pieces.setdefault(ti, set())
                cs.update(c - ta for c in (a, b) if ta < c < tb)
    split = {ti: piece_hyps(ref_tok[ti], ''.join(mapped[ti]) if ti < len(mapped) else '',
                            sorted(cs)) for ti, cs in pieces.items()}
    out = []
    for a, b, _, _ in segs:
        h = []
        for ti, (ta, tb) in enumerate(locs):
            if tb <= a or ta >= b:
                continue
            if a <= ta and tb <= b:
                h.extend(mapped[ti] if ti < len(mapped) else [])
            else:
                edges = [0] + sorted(pieces[ti]) + [tb - ta]
                k = edges.index(max(a, ta) - ta)
                if split[ti][k]:
                    h.append(split[ti][k])
        out.append(h)
    return out


def ref_to_hyp(ref_tokens, hyp_tokens):
    """-> per reference token, the hypothesis tokens aligned to it (possibly empty).

    jiwer gives chunk-level ops. A substitution or an equal run maps one-to-one. A deletion
    maps to nothing. An INSERTION has no reference token of its own, and an insertion beside
    a substitution is how the recogniser splitting one word in two shows up
    ('Netflix' -> 'net flicks': substitute + insert), so it is attached to the reference
    token it touches rather than thrown away -- otherwise that word scores as a wholesale
    miss when the take very nearly said it.
    """
    from jiwer import process_words
    out = [[] for _ in ref_tokens]
    ops = process_words(' '.join(ref_tokens), ' '.join(hyp_tokens)).alignments[0]
    for c in ops:
        if c.type in ('equal', 'substitute'):
            n = c.ref_end_idx - c.ref_start_idx
            for k in range(n):
                hi = c.hyp_start_idx + k
                if hi < c.hyp_end_idx and c.ref_start_idx + k < len(out):
                    out[c.ref_start_idx + k].append(hyp_tokens[hi])
            # a substitute chunk can be longer on the hypothesis side; the tail belongs to
            # the last reference token it replaced
            for hi in range(c.hyp_start_idx + n, c.hyp_end_idx):
                i = min(c.ref_end_idx - 1, len(out) - 1)
                if i >= 0:
                    out[i].append(hyp_tokens[hi])
        elif c.type == 'insert':
            # attach to the reference token on the left, which is where a split word's
            # second half sits; at the very start, attach to the first token instead
            i = c.ref_start_idx - 1 if c.ref_start_idx > 0 else 0
            i = min(i, len(out) - 1)
            if i >= 0:
                for hi in range(c.hyp_start_idx, c.hyp_end_idx):
                    out[i].append(hyp_tokens[hi])
    return out


def seg_score(ref_toks, hyp_toks, cross_script='zero'):
    """max(0, 1 - charCER) between a segment's reference and what was recognised for it.

    cross_script: a Latin reference rendered in Hangul/CJK ('series' -> '시리즈') has no
    meaningful character distance. 'zero' scores it 0, which is correct as a verdict but
    gives two such takes the SAME score and so no spread between them. 'romanize' falls back
    to a rough transliteration so 시리즈 and 씨리즈 can still be ordered.
    """
    ref = ''.join(ref_toks)
    hyp = ''.join(hyp_toks)
    if not ref:
        return None
    if not hyp:
        return 0.0
    r_cjk, h_cjk = is_cjk(ref[0]), is_cjk(hyp[0])
    if r_cjk != h_cjk:
        if cross_script == 'zero':
            return 0.0
        hyp = _romanize(hyp) if h_cjk else hyp
        ref = _romanize(ref) if r_cjk else ref
    from rapidfuzz.distance import Levenshtein
    return float(max(0.0, 1.0 - Levenshtein.distance(ref, hyp) / len(ref)))


_ROM = None


def _romanize(s):
    """Hangul -> rough latin, for cross_script='romanize'. Falls back to the input."""
    global _ROM
    if _ROM is None:
        try:
            from korean_romanizer.romanizer import Romanizer
            _ROM = lambda t: Romanizer(t).romanize()          # noqa: E731
        except Exception:
            _ROM = lambda t: t                                # noqa: E731
    try:
        return _ROM(s).lower()
    except Exception:
        return s


class CSSegmentReward(CSMerReward):
    """Per-segment CER scores plus each segment's time span in the rollout's audio.

    Returns the usual scalars so the executor's logging keeps working, and adds
    seg_scores / seg_spans / n_seg, which the segment-advantage executor consumes.
    """

    wants_segments = True
    wants_utt = True                   # the step-0 gate is keyed by utterance

    def __init__(self, *a, seg_unit='chunk', seg_chunk_chars=3, cross_script='zero',
                 gate=None, bpe_tokenizer=None, mos_backend=None, mos_lo=1.0, mos_hi=5.0,
                 utmosv2_repo='',
                 utmosv2_config='fusion_stage3', utmosv2_fold=0, **kw):
        super().__init__(*a, **kw)
        # Whole-utterance naturalness, reported as r_mos and rescaled to 0..1. The segment
        # executor turns it into a SECOND advantage that lands on every token, next to the
        # per-segment one -- the two measure different things and are kept apart rather than
        # summed into one score. None = no MOS model is loaded at all.
        self.mos = None
        self.mos_lo, self.mos_hi = mos_lo, mos_hi
        if mos_backend == 'utmosv2':
            self.mos = UtmosV2(device=self.device, repo=utmosv2_repo,
                               config=utmosv2_config, fold=utmosv2_fold)
        elif mos_backend == 'utmos22':
            self.mos = Utmos22(device=self.device)
        elif mos_backend not in (None, 'none'):
            raise ValueError(f'mos_backend must be utmos22, utmosv2 or none: {mos_backend}')
        if seg_unit not in ('cs', 'switch', 'chunk', 'mer', 'bpe'):
            raise ValueError("seg_unit must be 'cs', 'switch', 'chunk', 'mer' or 'bpe'")
        self.seg_unit, self.seg_chunk = seg_unit, seg_chunk_chars
        # seg_unit='bpe': the POLICY's text tokenizer -- a path, or the tokenizer object
        # itself (the SparkTTS trainer hands over the one its policy already loaded)
        self.bpe_tok = None
        if seg_unit == 'bpe':
            if bpe_tokenizer is None:
                bpe_tokenizer = BPE_TOKENIZER
            if isinstance(bpe_tokenizer, str):
                from transformers import AutoTokenizer
                bpe_tokenizer = AutoTokenizer.from_pretrained(bpe_tokenizer)
            self.bpe_tok = bpe_tokenizer
        self.cross_script = cross_script
        # step 0: segments the recogniser already fails on the GROUND TRUTH audio are a
        # scorer limitation, not a synthesis fault, and training on them optimises the ASR's
        # idiosyncrasies. build_seg_gate.py writes {utt: [keep_flag per segment]}.
        self.gate = None
        if gate:
            d = torch.load(gate, map_location='cpu', weights_only=False)
            # the gate indexes segments, so it is only valid for the segmentation it was
            # built with -- a silent mismatch would gate the wrong segments
            if d.get('seg_unit') != seg_unit or (
                    seg_unit == 'chunk' and d.get('seg_chunk_chars') != seg_chunk_chars):
                raise ValueError(
                    f'{gate} was built with seg_unit={d.get("seg_unit")} '
                    f'chunk={d.get("seg_chunk_chars")}, this reward uses '
                    f'{seg_unit}/{seg_chunk_chars}. Rebuild it with build_seg_gate.py.')
            self.gate = d['gate']
            logging.info('segment gate: %d utterances from %s', len(self.gate), gate)
        from transformers import GenerationConfig
        heads = getattr(self.model.generation_config, 'alignment_heads', None)
        if not heads:
            heads = GenerationConfig.from_pretrained(
                'openai/whisper-large-v3').alignment_heads
        self.heads = [(int(l), int(h)) for l, h in heads]
        self.sec_per_frame = 0.02

    # ------------------------------------------------------------------ timing
    @torch.inference_mode()
    def segment_spans(self, audio16k, text, code, char_ranges, interpolate=False):
        """-> [(t0, t1) or None] for each (char_start, char_end) of `text` in this audio.

        Same forced alignment Aligner.word_spans does -- teacher-force the reference, read
        the cross-attention, DTW the monotonic path -- but over arbitrary character ranges
        instead of its own word split. Raw attention spans OVERLAP (build_audio_refs.py
        measured 'climbing' 6.24-7.58 swallowing 'destination' 6.82-7.72), which for a
        per-segment mask would leak a neighbour's advantage onto this segment's tokens, so
        each span is then clipped to its neighbours.

        interpolate=True (seg_unit 'bpe'): a range that covers only PART of a Whisper token
        gets the matching fraction of that token's time, by character position. Without it
        two BPE pieces of one word ('determin' / 'ism') under one Whisper token would both
        get the whole word's span, and the neighbour clipping would then leave the second
        with nothing. For ranges made of whole Whisper tokens the result is the same.
        """
        prefix = self.tok.convert_tokens_to_ids(
            ['<|startoftranscript|>', f'<|{code}|>', '<|transcribe|>', '<|notimestamps|>'])
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        body, offs = list(enc.input_ids), enc.offset_mapping
        eot = self.tok.convert_tokens_to_ids('<|endoftext|>')
        if len(prefix) + len(body) + 1 > 440:
            return [None] * len(char_ranges)
        feats = self.fe(audio16k.cpu().numpy(), sampling_rate=16000,
                        return_tensors='pt').input_features
        out = self.model(input_features=feats.to(self.device, self.dt),
                         decoder_input_ids=torch.tensor(
                             [prefix + body + [eot]], device=self.device),
                         output_attentions=True)
        att = torch.stack([out.cross_attentions[l][0, h] for l, h in self.heads]).float()
        n_fr = min(att.shape[-1], int(len(audio16k) / 16000 / self.sec_per_frame) + 1)
        att = att[:, :, :n_fr]
        att = (att - att.mean(-1, keepdim=True)) / (att.std(-1, keepdim=True) + 1e-8)
        att = median_filter(att, 7).mean(0)
        rows = att[len(prefix) - 1:len(prefix) - 1 + len(body)]
        path = dtw(-rows.cpu().numpy().astype(np.float32))
        first, last = {}, {}
        for ti, fi in path:
            first.setdefault(ti, fi)
            last[ti] = fi
        raw = []
        for a, b in char_ranges:
            sel = [i for i, (ta, tb) in enumerate(offs) if ta < b and tb > a and i in first]
            if not sel:
                raw.append(None)
                continue
            if not interpolate:
                raw.append([min(first[i] for i in sel) * self.sec_per_frame,
                            (max(last[i] for i in sel) + 1) * self.sec_per_frame])
                continue
            t0s, t1s = [], []
            for i in sel:
                ta, tb = offs[i]
                while ta < tb and text[ta] == ' ':         # a token's leading space
                    ta += 1
                f0, f1 = first[i], last[i] + 1
                if tb > ta:
                    u0 = (max(a, ta) - ta) / (tb - ta)
                    u1 = (min(b, tb) - ta) / (tb - ta)
                else:
                    u0, u1 = 0.0, 1.0
                t0s.append(f0 + max(0.0, u0) * (f1 - f0))
                t1s.append(f0 + min(1.0, u1) * (f1 - f0))
            raw.append([min(t0s) * self.sec_per_frame, max(t1s) * self.sec_per_frame])
        # clip each span out of its neighbours' territory
        for i, sp in enumerate(raw):
            if sp is None:
                continue
            prev = next((raw[j] for j in range(i - 1, -1, -1) if raw[j]), None)
            nxt = next((raw[j] for j in range(i + 1, len(raw)) if raw[j]), None)
            if prev:
                sp[0] = max(sp[0], prev[1])
            if nxt and nxt[0] > sp[0]:
                sp[1] = min(sp[1], nxt[0])
        return [tuple(sp) if sp and sp[1] - sp[0] > 0.04 else None for sp in raw]

    # ------------------------------------------------------------------ scoring
    def __call__(self, wav, sr, gt_text, lang=None, ref_wav=None, ref_sr=None, utt=None):
        out = super().__call__(wav, sr, gt_text, lang, ref_wav, ref_sr)
        g16 = self._to16k(wav, sr)
        if self.mos is not None:
            try:
                raw = self.mos(g16)
                out['mos_raw'] = raw
                out['r_mos'] = rescale(raw, self.mos_lo, self.mos_hi)
            except Exception:
                out['mos_raw'], out['r_mos'] = None, None
        code = WHISPER_LANG.get(parse_lang(lang), 'en') if lang else 'en'
        ref_n = normalise(gt_text, code)
        hyp_n = normalise(out.get('hyp') or '', code)
        segs = ref_segments(ref_n, self.seg_unit, self.seg_chunk, self.bpe_tok,
                            raw_text=gt_text)
        out['n_seg'] = len(segs)
        if not segs:
            out['seg_scores'], out['seg_spans'], out['seg_kind'] = [], [], []
            return out
        ref_tok = tokenize_cs(ref_n)
        mapped = ref_to_hyp(ref_tok, tokenize_cs(hyp_n))
        # tokenize_cs already peels a Korean particle off an English word
        # ('series를' -> ['series', '를']), so no separate step is needed for it
        if self.seg_unit == 'bpe':
            # a segment may be a piece of a word, which the token walk below cannot express
            per_seg = [(toks, h) for (_, _, toks, _), h in
                       zip(segs, segment_hyps(ref_n, ref_tok, mapped, segs))]
        else:
            idx, per_seg = 0, []
            for _, _, toks, _ in segs:
                n = len(toks)
                hyp_toks = [h for k in range(idx, idx + n) if k < len(mapped)
                            for h in mapped[k]]
                per_seg.append((toks, hyp_toks))
                idx += n
        if self.seg_unit == 'cs':
            # pool every run of the same language into one score, then give that score back
            # to each of its runs. The executor normalises per segment index, so identical
            # scores produce identical advantages and the whole language moves as one.
            pooled = {}
            for kind in ('en', 'matrix'):
                ks = [i for i, sg in enumerate(segs) if sg[3] == kind]
                if not ks:
                    continue
                ref_t = [t for i in ks for t in per_seg[i][0]]
                hyp_t = [h for i in ks for h in per_seg[i][1]]
                pooled[kind] = seg_score(ref_t, hyp_t, self.cross_script)
            scores = [pooled.get(sg[3]) for sg in segs]
        else:
            scores = [seg_score(t, h, self.cross_script) for t, h in per_seg]
        keep = self.gate.get(utt) if (self.gate is not None and utt is not None) else None
        if keep is not None:
            scores = [s if (i < len(keep) and keep[i]) else None
                      for i, s in enumerate(scores)]
        out['seg_scores'] = scores
        out['seg_kind'] = [k for _, _, _, k in segs]
        out['seg_spans'] = self.segment_spans(g16, ref_n, code,
                                              [(a, b) for a, b, _, _ in segs],
                                              interpolate=(self.seg_unit == 'bpe'))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lst', required=True)
    ap.add_argument('--wav_dir', required=True)
    ap.add_argument('--model', default=os.path.join(WFT, 'exp/whisper_lora/merged'))
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--seg_unit', default='chunk',
                    choices=['cs', 'switch', 'chunk', 'mer', 'bpe'])
    ap.add_argument('--seg_chunk_chars', type=int, default=3)
    ap.add_argument('--bpe_tokenizer', default=None, help="seg_unit='bpe' only")
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--lang', default=None,
                    help='matrix language; default: the .lst basename')
    ap.add_argument('--limit', type=int, default=3)
    args = ap.parse_args()
    # read_lst carries no language field, and CSMerReward forces the decoder
    # language from it -- left as None the matrix half is TRANSLATED into English
    # and every native segment scores 0. The testset files are named after their
    # language, which is the reliable source.
    lang = args.lang or os.path.splitext(os.path.basename(args.lst))[0]
    R = CSSegmentReward(model=args.model, adapter=args.adapter, device=args.device,
                        seg_unit=args.seg_unit, seg_chunk_chars=args.seg_chunk_chars,
                        bpe_tokenizer=args.bpe_tokenizer)
    for r in read_lst(args.lst)[:args.limit]:
        wav = os.path.join(args.wav_dir, r['utt_id'] + '.wav')
        if not os.path.exists(wav):
            continue
        o = R(wav, None, r['text'], lang, utt=r['utt_id'])
        print(f"\n[{r['utt_id']}] MER={o['mer']:.3f}  segments={o['n_seg']}")
        print(f"  HYP: {o['hyp'][:100]}")
        code = WHISPER_LANG.get(parse_lang(lang), 'en')
        segs = ref_segments(normalise(r['text'], code), args.seg_unit, args.seg_chunk_chars,
                            R.bpe_tok, raw_text=r['text'])
        for (a, b, toks, kind), s, sp in zip(segs, o['seg_scores'], o['seg_spans']):
            t = '-' if sp is None else f'{sp[0]:.2f}-{sp[1]:.2f}s'
            tk = '-' if sp is None else f'tok {int(sp[0] * TOKEN_HZ)}..{int(sp[1] * TOKEN_HZ)}'
            print(f"    {kind:6s} {''.join(toks):22s} score={('-' if s is None else f'{s:.3f}')}"
                  f"  {t:16s} {tk}")


if __name__ == '__main__':
    main()
