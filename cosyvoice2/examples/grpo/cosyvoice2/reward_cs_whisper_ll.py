# GRPO reward #3 -- teacher-forced Whisper log-likelihood. No decoding, no alignment.
#
# reward_cs_wer.py asks "what does the ASR write?" and then compares. That question is the
# source of every code-switching failure we hit: free decoding lets the ASR decide which
# LANGUAGE it is hearing, so a Korean-embedded 'basically' comes back as '베이직리' and the
# WER cannot tell a good English realisation from a Hangul-ised one.
#
# But in TTS-RL the target text is KNOWN. So do not ask what the ASR writes -- ask how
# likely the reference text is, given the audio:
#
#     R = 1/T * sum_t log p_theta(y_t | y_<t, audio)
#
# One encoder+decoder forward with the reference tokens teacher-forced. The three problems
# of the alignment pipeline dissolve:
#   * language mis-ID     -> the reference tokens are forced, the model never chooses a script
#   * CTC blank / silence -> Whisper is seq2seq, there is no blank and no alignment step
#   * unromanizable text  -> log-probs are taken on native-script tokens; uroman is gone,
#                            so the Japanese-kanji failure of cs_reward_common.py disappears
#
# Per-language decomposition survives: every reference token carries a character offset, so
# tokens are labelled 'en' vs 'matrix' by script and averaged separately. R_embedded is
# still the term that watches the English words.
#
# The score is a mean log-prob in (-inf, 0]. It is reported as exp(mean log-prob), i.e. the
# geometric-mean per-token probability, which lands in (0, 1] and is directly mixable with
# the speaker/duration terms in cs_reward_common.py.
#
# Offline check on one file list:
#   python reward_cs_whisper_ll.py --lst .../Korean.lst --wav_dir <results>/Korean/<model> --limit 20 -v
# Per-language sanity sweep (this is the one to run first):
#   python check_whisper_ll_langs.py --help
from __future__ import annotations

import argparse
import re
import warnings

import numpy as np
import torch
import torchaudio

from cs_reward_common import (SpeakerSim, combine, duration_reward, is_cjk,
                              mix_all)

CAMPPLUS = ''

# corpus label (lst stem or the `lang` column) -> Whisper language token.
# Whisper large-v3 added <|yue|>, so Cantonese no longer has to borrow <|zh|>.
WHISPER_LANG = {
    'Korean': 'ko', 'kor-eng': 'ko', 'kor': 'ko', 'ko': 'ko',
    'Mandarin': 'zh', 'cmn-eng': 'zh', 'cmn': 'zh', 'zh': 'zh',
    'Cantonese': 'yue', 'yue-eng': 'yue', 'yue': 'yue',
    'Japanese': 'ja', 'jpn-eng': 'ja', 'jpn': 'ja', 'ja': 'ja',
    'Arabic': 'ar', 'ara-eng': 'ar', 'ara': 'ar', 'ar': 'ar',
    'French': 'fr', 'fra-eng': 'fr', 'fra': 'fr', 'fr': 'fr',
    'German': 'de', 'deu-eng': 'de', 'deu': 'de', 'de': 'de',
    'Hindi': 'hi', 'hin-eng': 'hi', 'hin': 'hi', 'hi': 'hi',
    'Italian': 'it', 'ita-eng': 'it', 'ita': 'it', 'it': 'it',
    'Russian': 'ru', 'rus-eng': 'ru', 'rus': 'ru', 'ru': 'ru',
    'Spanish': 'es', 'spa-eng': 'es', 'spa': 'es', 'es': 'es',
    'Czech': 'cs', 'ces-eng': 'cs', 'ces': 'cs', 'cs': 'cs',
    'English': 'en', 'eng': 'en', 'en': 'en',
}

_LATIN_CH = re.compile(r"[A-Za-z']")
# Latin letters INCLUDING diacritics, so 'santé' and 'Mischung' are one word each and get
# one language label. Without the accented ranges 'sant'/'é' would split across buckets.
_LATIN_WORD = re.compile(r"[A-Za-z\u00C0-\u024F][A-Za-z\u00C0-\u024F'\-]*")


def nonlatin_script(ch):
    """True for a letter of a script that is not Latin (Cyrillic, Arabic, Devanagari,
    Hangul, Kana, Han, ...). An accented Latin letter is NOT one: 'é' must not make a
    French sentence look like a non-Latin language."""
    return ch.isalpha() and ord(ch) > 0x02FF


def parse_lang(s):
    """Corpus language label out of whatever the training loop has: a label already
    ('Korean'), or an utterance id ('Korean_102_0', 'csf_read_test_kor_1662_KC').

    The executor used to take utt_id.split('_')[0], which is 'csf' for every cs-fleurs
    utterance -- so the whole cs-fleurs half of the corpus fell back to script guessing,
    and script guessing cannot tell Mandarin from Cantonese.
    """
    s = str(s)
    if s in WHISPER_LANG:
        return s
    for part in re.split(r'[^A-Za-z]+', s):
        if part in WHISPER_LANG:
            return part
        if part.lower() in WHISPER_LANG:
            return part.lower()
    return s.split('_')[0]


def whisper_lang(lang, text=None, default='en'):
    if lang in WHISPER_LANG:
        return WHISPER_LANG[lang]
    if lang and lang.lower() in WHISPER_LANG:
        return WHISPER_LANG[lang.lower()]
    if text:
        from cs_reward_common import guess_lang
        return guess_lang(text)
    return default


def char_kinds(text, en_words=None):
    """-> list, one entry per character: 'en' | 'matrix' | None.

    None is punctuation, whitespace and digits: they belong to neither language, so they
    are counted in R_all but excluded from R_matrix / R_embedded.

    By default the split is by script, which is exact whenever the matrix language is not
    written in Latin (ko, zh, yue, ja, ar, ru, ...). For a Latin-script matrix language
    (de, fr, it, es, cs, and the romanized Hindi in this corpus) script tells you nothing --
    'Legierungen' would be labelled English. Pass `en_words`, a set of lowercased English
    words, and Latin words are then bucketed by membership instead. Without it, a Latin
    matrix language collapses into R_embedded and only R_all is meaningful.
    """
    out = [None] * len(text)
    for i, ch in enumerate(text):
        if is_cjk(ch) or (ch.isalpha() and not ch.isascii()):
            out[i] = 'matrix'
    for m in _LATIN_WORD.finditer(text):
        kind = 'en'
        if en_words is not None:
            kind = 'en' if m.group(0).strip("'-").lower() in en_words else 'matrix'
        for i in range(m.start(), m.end()):
            out[i] = kind
    return out


def latin_matrix(texts, thresh=0.10):
    """True when a language's own script is Latin, i.e. the script split cannot separate
    the embedded English. Measured as the share of letters belonging to a non-Latin script:
    'Legierungen sind basically...' scores 0.0, Korean text ~0.8."""
    n = nz = 0
    for t in texts:
        for ch in t:
            if ch.isalpha():
                n += 1
                nz += nonlatin_script(ch)
    return (nz / n) < thresh if n else False


class WhisperLL:
    """Teacher-forced Whisper scorer. One forward per utterance, batched.

    score(wavs, texts, langs) -> list of per-utterance dicts with the raw mean log-probs
    and the per-token detail. Everything above it (reward shaping, mixing) is separate so
    the validation driver can look at the raw numbers.
    """

    def __init__(self, model='openai/whisper-large-v3', device='cuda:0', dtype='float16',
                 leading_space=True, score_eot=True):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.device = device
        self.dtype = dict(float16=torch.float16, bfloat16=torch.bfloat16,
                          float32=torch.float32)[dtype]
        if not str(device).startswith('cuda'):
            self.dtype = torch.float32
        self.proc = WhisperProcessor.from_pretrained(model)
        self.tok = self.proc.tokenizer
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model, torch_dtype=self.dtype).to(device).eval()
        self.sr = self.proc.feature_extractor.sampling_rate
        # Whisper only ever sees a 30s window; anything past it is silently dropped and its
        # reference tokens would be scored against nothing.
        self.max_sec = (self.proc.feature_extractor.n_samples / self.sr)
        self.leading_space = leading_space
        self.score_eot = score_eot
        self._sot = self.tok.convert_tokens_to_ids('<|startoftranscript|>')
        self._task = self.tok.convert_tokens_to_ids('<|transcribe|>')
        self._nots = self.tok.convert_tokens_to_ids('<|notimestamps|>')
        self._eot = self.tok.convert_tokens_to_ids('<|endoftext|>')
        self._unk_langs = set()

    # ---------------------------------------------------------------- tokenisation
    def _lang_id(self, lang):
        tid = self.tok.convert_tokens_to_ids(f'<|{lang}|>')
        if tid is None or tid == self.tok.unk_token_id:
            if lang not in self._unk_langs:
                warnings.warn(f'no Whisper token <|{lang}|>, falling back to <|en|>')
                self._unk_langs.add(lang)
            tid = self.tok.convert_tokens_to_ids('<|en|>')
        return tid

    def _encode(self, text):
        """-> (ids, [(char_start, char_end) per id]).

        The offsets are what makes the per-language decomposition possible: a token is
        labelled by the characters it covers, so a Hangul token and the English token next
        to it end up in different buckets even inside one whitespace word.
        """
        ids = self.tok(text, add_special_tokens=False).input_ids
        offs = self._offsets_fast(text, ids)
        if offs is None:
            offs = self._offsets_bytes(text, ids)
        return ids, offs

    def _offsets_fast(self, text, ids):
        if not getattr(self.tok, 'is_fast', False):
            return None
        try:
            enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        except Exception:
            return None
        if list(enc.input_ids) != list(ids):
            return None
        offs = [tuple(o) for o in enc.offset_mapping]
        if not offs or all(a == b for a, b in offs):
            return None   # degenerate mapping, use the byte path instead
        return offs

    def _offsets_bytes(self, text, ids):
        """Byte-level BPE fallback: token strings decode back to exact utf-8 bytes."""
        bd = getattr(self.tok, 'byte_decoder', None)
        raw = text.encode('utf-8')
        b2c = []
        for ci, ch in enumerate(text):
            b2c.extend([ci] * len(ch.encode('utf-8')))
        if bd is None:
            # last resort: spread the tokens uniformly over the characters
            n = max(1, len(ids))
            step = len(text) / n
            return [(int(i * step), int((i + 1) * step)) for i in range(n)]
        offs, pos = [], 0
        for t in self.tok.convert_ids_to_tokens(ids):
            try:
                nb = len(bytes(bd[c] for c in t))
            except KeyError:
                nb = len(t.encode('utf-8'))
            a = b2c[pos] if pos < len(b2c) else len(text)
            end = min(pos + nb, len(raw))
            b = (b2c[end - 1] + 1) if end > 0 and end <= len(b2c) else len(text)
            offs.append((a, max(a, b)))
            pos += nb
        return offs

    # ---------------------------------------------------------------- scoring
    @torch.inference_mode()
    def score(self, wavs, texts, langs, sr=16000, en_words=None, ids_list=None):
        """wavs: list of 1-D float tensors at `sr`. -> list of dicts.

        dict = {lp_all, lp_matrix, lp_en, lp_*_body, lp_eot, n_*, tokens, clipped}
        ids_list: optional explicit token ids per utterance, in place of self._encode.
        lp_*_body excludes the first token of each language span (the switch point).
        `tokens` is [(surface, kind, logprob, is_span_onset)].
        en_words: optional lowercased English lexicon, needed only for Latin-script matrix
        languages -- see char_kinds().
        where lp_* are MEAN log-probs (nats/token) and `tokens` is
        [(surface, kind, logprob)] for inspection.
        """
        assert len(wavs) == len(texts) == len(langs)
        proc_wavs = []
        clipped = []
        for w in wavs:
            if w.dim() > 1:
                w = w.mean(dim=0)
            clipped.append(bool(w.numel() > self.max_sec * sr))
            proc_wavs.append(w.float().cpu().numpy())
        feats = self.proc.feature_extractor(proc_wavs, sampling_rate=sr,
                                            return_tensors='pt').input_features

        prefixes, targets, offsets, used = [], [], [], []
        for n, (text, lang) in enumerate(zip(texts, langs)):
            t = (' ' + text.strip()) if self.leading_space else text.strip()
            if ids_list is None:
                ids, offs = self._encode(t)
            else:
                # an explicit segmentation of the SAME text. Byte-level BPE means many token
                # sequences decode to one string, and only one of them is what _encode picks;
                # check_whisper_tokenization.py uses this to ask how much the score depends
                # on which one is forced.
                ids = list(ids_list[n])
                offs = self._offsets_bytes(t, ids)
            prefixes.append([self._sot, self._lang_id(lang), self._task, self._nots])
            targets.append(ids)
            offsets.append(offs)
            used.append(t)

        P = len(prefixes[0])
        Lmax = max(len(t) for t in targets) + 1          # +1 for the eot position
        dec = torch.full((len(targets), P + Lmax - 1), self._eot, dtype=torch.long)
        for i, (pre, tgt) in enumerate(zip(prefixes, targets)):
            row = pre + tgt
            dec[i, :len(row)] = torch.tensor(row)
        # right padding is harmless: the decoder is causal, so a padded position can only
        # affect positions after it, which are never read.

        out = self.model(input_features=feats.to(self.device, self.dtype),
                         decoder_input_ids=dec.to(self.device))
        logprobs = torch.log_softmax(out.logits.float(), dim=-1).cpu()

        results = []
        for i, (tgt, offs, text) in enumerate(zip(targets, offsets, used)):
            kinds = char_kinds(text, en_words)
            lps = [float(logprobs[i, P - 1 + j, tid]) for j, tid in enumerate(tgt)]
            lp_eot = float(logprobs[i, P - 1 + len(tgt), self._eot])
            det, acc = [], {'en': [], 'matrix': []}
            body = {'en': [], 'matrix': []}
            prev = None
            for j, (tid, (a, b)) in enumerate(zip(tgt, offs)):
                seg = [k for k in kinds[a:b] if k]
                kind, onset = None, False
                if seg:
                    kind = 'en' if seg.count('en') >= seg.count('matrix') else 'matrix'
                    # The FIRST token of a language span pays the language model's
                    # switch-point surprise, not a pronunciation cost: on a Korean
                    # utterance the ' It' that opens an English span scored -6.16 while
                    # every following English token scored above -0.44. Keep it in lp_en
                    # but also report the span body without it, so the caller can choose
                    # whether the reward should include "was the switch predictable".
                    onset = (kind != prev)
                    acc[kind].append(lps[j])
                    if not onset:
                        body[kind].append(lps[j])
                prev = kind if kind else prev
                det.append((text[a:b], kind, lps[j], onset))
            mean = lambda v: float(np.mean(v)) if v else None
            all_lps = lps + ([lp_eot] if self.score_eot else [])
            results.append({
                'lp_all': float(np.mean(all_lps)) if all_lps else float('nan'),
                'lp_matrix': mean(acc['matrix']), 'lp_en': mean(acc['en']),
                'lp_matrix_body': mean(body['matrix']), 'lp_en_body': mean(body['en']),
                'lp_eot': lp_eot,
                'n_all': len(tgt), 'n_matrix': len(acc['matrix']), 'n_en': len(acc['en']),
                'n_en_body': len(body['en']), 'n_matrix_body': len(body['matrix']),
                'tokens': det, 'clipped': clipped[i],
            })
        return results


def lp_to_r(lp):
    """mean log-prob (nats/token) -> geometric-mean token probability in (0, 1]."""
    return None if lp is None else float(np.exp(lp))


class CSWhisperLLReward:
    """Drop-in sibling of CSWerReward / CSAlignReward.

    __call__(wav, sr, gt_text, lang, ref_wav, ref_sr) -> dict(reward, r_content,
    r_matrix, r_embedded, r_spk, r_dur, ...). Same keys, so the GRPO glue does not care
    which of the three rewards it was handed.
    """

    def __init__(self, model='openai/whisper-large-v3', device='cuda:0', dtype='float16',
                 alpha=1.0, beta=2.0, w_content=1.0, w_spk=0.0, w_dur=0.0,
                 campplus_path=CAMPPLUS, score_eot=True, en_words=None,
                 skip_span_onset=False):
        self.ll = WhisperLL(model=model, device=device, dtype=dtype, score_eot=score_eot)
        self.alpha, self.beta = alpha, beta
        self.w_content, self.w_spk, self.w_dur = w_content, w_spk, w_dur
        self.en_words = en_words
        # True: R_matrix / R_embedded ignore each span's first token, so the reward asks
        # only "was this pronounced well", not "was the code switch predictable".
        self.skip_span_onset = skip_span_onset
        self.spk = SpeakerSim(campplus_path, device=device) if w_spk > 0 else None

    def _to16k(self, wav, sr):
        if isinstance(wav, str):
            wav, sr = torchaudio.load(wav)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        return wav.squeeze(0)

    def batch(self, wavs, srs, texts, langs):
        w16 = [self._to16k(w, s) for w, s in zip(wavs, srs)]
        return self.ll.score(w16, texts, [whisper_lang(l, t) for l, t in zip(langs, texts)],
                             en_words=self.en_words)

    def _finish(self, res, wav, sr, ref_wav=None, ref_sr=None):
        sfx = '_body' if self.skip_span_onset else ''
        r_m = lp_to_r(res['lp_matrix' + sfx] if res['lp_matrix' + sfx] is not None
                      else res['lp_matrix'])
        r_e = lp_to_r(res['lp_en' + sfx] if res['lp_en' + sfx] is not None else res['lp_en'])
        r_content = combine(r_m, r_e, self.alpha, self.beta)
        r_spk = r_dur = None
        if ref_wav is not None:
            g = self._to16k(wav, sr)
            r = self._to16k(ref_wav, ref_sr or 16000)
            r_dur = duration_reward(g.numel() / 16000, r.numel() / 16000)
            if self.spk is not None:
                r_spk = self.spk(g.unsqueeze(0), r.unsqueeze(0))
        return {'reward': mix_all(r_content, r_spk, r_dur,
                                  self.w_content, self.w_spk, self.w_dur),
                'r_content': r_content, 'r_matrix': r_m, 'r_embedded': r_e,
                'r_all': lp_to_r(res['lp_all']), 'r_spk': r_spk, 'r_dur': r_dur,
                'lp': res}

    def score_many(self, wavs, srs, texts, langs, ref_wavs=None, ref_srs=None):
        """Same output as __call__, one dict per utterance, in ONE Whisper forward.

        A GRPO group is `rollout_n` takes of the same sentence, so they batch perfectly:
        4 rollouts cost one padded 30s forward instead of four. The per-rollout ASR call
        that reward_cs_wer.py needs has no such option -- this is the reward's other
        throughput advantage, on top of not decoding.
        """
        res = self.batch(wavs, srs, texts, langs)
        n = len(wavs)
        ref_wavs = ref_wavs or [None] * n
        ref_srs = ref_srs or [None] * n
        return [self._finish(r, w, s, rw, rs)
                for r, w, s, rw, rs in zip(res, wavs, srs, ref_wavs, ref_srs)]

    def __call__(self, wav, sr, gt_text, lang=None, ref_wav=None, ref_sr=None):
        return self.score_many([wav], [sr], [gt_text], [lang], [ref_wav], [ref_sr])[0]


def read_lst(path):
    rows = []
    for line in open(path, encoding='utf-8'):
        p = line.rstrip('\n').split('|')
        if len(p) >= 4:
            rows.append({'utt_id': p[0], 'prompt_text': p[1], 'prompt_wav': p[2],
                         'text': p[3], 'gt_wav': p[4] if len(p) > 4 else None})
    return rows


def main():
    ap = argparse.ArgumentParser(description='Offline check of the teacher-forced Whisper LL reward')
    ap.add_argument('--lst', required=True)
    ap.add_argument('--wav_dir', default=None,
                    help='default: score the lst audio itself (ground truth topline)')
    ap.add_argument('--lang', default=None, help='e.g. Korean / kor-eng; default: lst stem')
    ap.add_argument('--model', default='openai/whisper-large-v3')
    ap.add_argument('--limit', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--dtype', default='float16')
    ap.add_argument('-v', '--verbose', action='store_true', help='dump per-token log-probs')
    args = ap.parse_args()

    import os
    lang = args.lang or os.path.basename(args.lst).rsplit('.', 1)[0]
    rows = read_lst(args.lst)[:args.limit]
    rw = CSWhisperLLReward(model=args.model, device=args.device, dtype=args.dtype)

    R, RM, RE = [], [], []
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i:i + args.batch_size]
        wavs, srs, texts, keep = [], [], [], []
        for r in chunk:
            path = (os.path.join(args.wav_dir, r['utt_id'] + '.wav') if args.wav_dir
                    else (r['gt_wav'] or r['prompt_wav']))
            try:
                w, s = torchaudio.load(path)
            except Exception as e:
                print(f"skip {r['utt_id']}: {e}")
                continue
            wavs.append(w); srs.append(s); texts.append(r['text']); keep.append(r)
        if not wavs:
            continue
        res = rw.batch(wavs, srs, texts, [lang] * len(wavs))
        for r, out in zip(keep, res):
            r_m, r_e = lp_to_r(out['lp_matrix']), lp_to_r(out['lp_en'])
            r_c = combine(r_m, r_e, rw.alpha, rw.beta)
            R.append(r_c)
            if r_m is not None:
                RM.append(r_m)
            if r_e is not None:
                RE.append(r_e)
            f = lambda v: '-' if v is None else format(v, '.3f')
            print(f"[{r['utt_id']}] R={r_c:.3f} all={lp_to_r(out['lp_all']):.3f} "
                  f"matrix={f(r_m)} embedded={f(r_e)} "
                  f"lp_all={out['lp_all']:+.2f} n_en={out['n_en']}"
                  + ('  [CLIPPED >30s]' if out['clipped'] else ''))
            if args.verbose:
                for surface, kind, lp, onset in out['tokens']:
                    print(f"    {str(kind or '-'):<7} {lp:+7.3f}{' <-switch' if onset else '        '}"
                          f"  {surface!r}")
    if R:
        print(f"\nmean R {np.mean(R):.3f} | matrix {np.mean(RM) if RM else float('nan'):.3f}"
              f" | embedded {np.mean(RE) if RE else float('nan'):.3f} | n={len(R)}")


if __name__ == '__main__':
    main()
