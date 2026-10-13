# GRPO reward #7 -- decoded MER + UTMOS + speaker similarity, on one scale.
#
# The three numbers this project reports for every system are MER, UTMOS and SECS, and
# this reward optimises all three directly. Both extra scorers are the SAME models the
# evaluation uses, so "the reward went up" and "the table got better" are the same claim:
#
#   MER    whisper_ft/exp/whisper_lora/merged     (reward_cs_mer.CSMerReward, unchanged)
#   UTMOS  UTMOSv2 fusion_stage3                  == UTMOSv2/infer_cosyvoice.sh
#          or utmos22_strong (torch.hub SpeechMOS)   the fast stand-in, see COST below
#   SECS   microsoft/wavlm-base-plus-sv           == TTS-Evaluation run_wavlm_cs.sh
#                                                    (evaluation/sv_pipeline.py 'wavlm')
#
# The content half is delegated to CSMerReward rather than reimplemented, with w_spk and
# w_dur forced to 0 there: whatever MER does here is bit-identical to what it does in
# run_grpo_mer_sft.sh, so that run stays a clean ablation of this one.
#
# ---------------------------------------------------------------------------- COST
# Measured on this box, per rollout:
#
#   MER (whisper-large-v3 greedy decode)   ~1.0 s
#   utmos22_strong                         ~0.007 s      <- 4 in a batch: 26 ms total
#   UTMOSv2 fusion_stage3, 1 fold          ~0.53 s       <- (63.89-29.93)/64, n=32 vs 96
#   wavlm-base-plus-sv x-vector            ~0.02 s
#
# UTMOSv2 is 75x the cost of utmos22 and lands on the same axis; its melspec front end
# (n_fft 4096, hop 32, four resolutions, 512x512 each) is CPU-bound and per-file. At 4
# rollouts x ~2 items x 2681 steps that is ~3 h of added wall clock for the exact metric,
# versus ~2 min for the stand-in. Default is utmos22; set mos_backend: utmosv2 when the
# point is to optimise the reported number itself.
#
# ------------------------------------------------------------------- READ THIS FIRST
# UTMOS IS NOT A TOPLINE ON THIS DATA. From the eval table (sft_to_grpo_analysis.md 1-4):
#
#     Korean     GT 2.528   SFT 3.246        Japanese   GT 2.062   SFT 2.909
#     Mandarin   GT 2.550   SFT 3.107        Cantonese  GT 2.962   SFT 3.215
#
# The SFT model ALREADY scores far above the real recordings, on every subset. So
# maximising UTMOS moves the policy AWAY from human speech, and the direction it moves is
# the one UTMOS likes: clean, smooth, low-variance -- the same flat prosody the CS damage
# analysis measured as F0-range collapse. This is exactly the trap audio_reference_reward.md
# 7.3 describes for the native score, and it takes the same answer: use it as a FLOOR, not
# as an objective. mos_floor=<x> zeroes any rollout below x and adds nothing above it --
# "do not get worse" rather than "get smoother". w_mos > 0 is supported because it was
# asked for; prefer w_mos=0 with mos_floor set.
#
# ------------------------------------------------------------------- CONTENT SCOPE
# DEFAULT content_scope='all': the content term is the plain whole-utterance MER,
#
#     R_content = 1 - tanh(sharpness * MER)
#
# which is literally the number seed-tts-eval/cal_mer_whisper_ft.sh reports. No language
# split enters the objective.
#
# This is only possible because the scorer was fine-tuned. The old ASR could not read
# English at all (28.1% English loss on real human speech, 0.2% after LoRA), which is why
# reward_cs_wer.py had to crop every span and transcribe it under its own language tag --
# the split was a workaround for a broken scorer, not a modelling choice. One full-utterance
# decode now carries gradient for both languages.
#
# r_matrix and r_embedded are still computed on every rollout and still reported, but they
# are OBSERVED ONLY, the same way audio_reference_reward.md 7.3 treats the native phoneme
# score: in the reward they would re-weight the objective, on a curve they answer whether
# the two languages moved together.
#
# What to expect on those curves, since 'all' weights every TOKEN equally and English is a
# token minority everywhere (counted on the ground-truth references):
#
#     subset      tokens   English   English share of tokens
#     kor-eng      18628      2007            10.8%
#     jpn-eng       7896      1624            20.6%
#     cmn-eng      38694     12634            32.7%
#     Korean        2879       712            24.7%
#     Cantonese     6635      1482            22.3%
#     ALL          80400     19745            24.6%
#
# So on kor-eng roughly 89% of the content signal is the native language, which already
# reads correctly (matrix MER 7.1% against 26.5% overall). If r_embedded flattens or falls
# while r_all climbs, the run is buying its MER from the easy language -- that is the
# finding content_scope='split' (alpha 1 : beta 2) exists to test against, and it is one
# config line away.
#
# SECS has the opposite problem: it is pinned at 0.96-0.98 for every system, so its spread
# inside a group of 4 rollouts of ONE sentence may be ~0. A term that is constant inside a
# group is deleted by the group-mean subtraction whatever its weight. reward_std_spk in
# tensorboard is the only thing that settles it -- if it sits near 0, w_spk did nothing and
# spk_lo should be raised toward the cross-speaker floor to stretch the useful range.
#
# Offline check:
#   python reward_cs_mer_mos.py --lst <meta.lst> --wav_dir <results>/<system> --limit 20
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torchaudio

from reward_cs_mer import CSMerReward
from reward_cs_whisper_ll import read_lst

WAVLM_SV = 'microsoft/wavlm-base-plus-sv'
# TTS-Evaluation/evaluation/sv_pipeline.py pads to 2 s before embedding: WavLM's TDNN stack
# needs that much context and shorter clips give a meaningless cosine (measured against the
# full utterance: 0.5 s -> 0.20, 1 s -> 0.37, 2 s -> 0.85, 3 s -> 0.93). Same constant here.
SV_MIN_SAMPLES = 32000


class Utmos22:
    """torch.hub SpeechMOS utmos22_strong. One wav2vec2-base forward, batches natively."""

    name = 'utmos22'

    # torch.hub.load() on the github spec downloads and unzips into ~/.cache/torch/hub,
    # and all four DDP ranks build their reward at the same moment -- they would race on
    # the same extraction. Load from the already-extracted local checkout when it is
    # there, which makes construction read-only and rank-safe.
    HUB_CACHE = os.path.expanduser('~/.cache/torch/hub')

    def __init__(self, device='cuda:0'):
        self.device = device
        local = None
        for d in ('tarepan_SpeechMOS_main', 'tarepan_SpeechMOS_v1.2.0'):
            c = os.path.join(self.HUB_CACHE, d)
            if os.path.isdir(os.path.join(c, 'speechmos')):
                local = c
                break
        if local is not None:
            self.m = torch.hub.load(local, 'utmos22_strong', source='local',
                                    trust_repo=True)
        else:
            self.m = torch.hub.load('tarepan/SpeechMOS', 'utmos22_strong',
                                    trust_repo=True)
        self.m = self.m.to(device).eval()

    @torch.inference_mode()
    def __call__(self, wav16k):
        # utmos22 was trained on >=2 s; below that the score drifts low regardless of
        # quality (measured: 1.38 at 0.5 s vs 1.99 at 2 s on the same recording), so pad
        # rather than hand GRPO a length artefact dressed up as a quality score.
        if wav16k.numel() < SV_MIN_SAMPLES:
            wav16k = torch.nn.functional.pad(
                wav16k, (0, SV_MIN_SAMPLES - wav16k.numel()))
        return float(self.m(wav16k.unsqueeze(0).to(self.device), sr=16000))


class UtmosV2:
    """UTMOSv2 fusion_stage3 -- the model UTMOSv2/infer_cosyvoice.sh reports.

    predict(data=..., sr=...) takes a waveform directly, so no rollout has to be written
    to disk. num_workers=0 because this already runs inside a DDP worker: the default 4
    would fork a dataloader pool per rank on every single rollout.

    EXTRA DEPENDENCIES. utmosv2 imports timm (-> torchvision), which the cosyvoice env
    does not have. `pip install timm` there before setting mos_backend: utmosv2; the
    utmos22 backend needs nothing beyond torch.hub.
    """

    name = 'utmosv2'

    def __init__(self, device='cuda:0', repo='',
                 config='fusion_stage3', fold=0, tmpdir=None):
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from utmosv2 import create_model
        self.device = device
        self.model = create_model(config=config, fold=fold, device=device)

    @torch.inference_mode()
    def __call__(self, wav16k):
        v = self.model.predict(data=wav16k.cpu().numpy(), sr=16000,
                               device=self.device, num_workers=0, batch_size=1,
                               verbose=False)
        return float(np.asarray(v).mean())


class WavLMSpeakerSim:
    """Cosine between wavlm-base-plus-sv x-vectors, identical to the SECS column.

    TTS-Evaluation/evaluation/sv_pipeline.py: librosa at 16 kHz, zero-pad to 2 s, the
    HF feature extractor, then F.cosine_similarity on .embeddings[0]. The reference
    embedding is cached per prompt -- inside a group the prompt never changes, so this is
    one forward per rollout instead of two.
    """

    def __init__(self, device='cuda:0', model=WAVLM_SV):
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
        self.device = device
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(model)
        self.m = WavLMForXVector.from_pretrained(model).to(device).eval()
        self._ref = (None, None)

    @torch.inference_mode()
    def embed(self, wav16k):
        if wav16k.numel() < SV_MIN_SAMPLES:
            wav16k = torch.nn.functional.pad(
                wav16k, (0, SV_MIN_SAMPLES - wav16k.numel()))
        i = self.fe([wav16k.cpu().numpy()], padding=True, return_tensors='pt',
                    sampling_rate=16000)
        i = {k: v.to(self.device) for k, v in i.items()}
        return self.m(**i).embeddings[0]

    def __call__(self, gen16k, ref16k, ref_key=None):
        if ref_key is not None and self._ref[0] == ref_key:
            e_ref = self._ref[1]
        else:
            e_ref = self.embed(ref16k)
            self._ref = (ref_key, e_ref)
        cos = torch.nn.functional.cosine_similarity(self.embed(gen16k), e_ref, dim=-1)
        return float(cos)


def rescale(v, lo, hi):
    """Affine map onto [0, 1]. GRPO standardises advantages inside a group, so lo/hi
    cannot change the ranking -- but they do set how much SPREAD this term contributes
    relative to the others, which is what the weights are then balancing."""
    if v is None:
        return None
    if hi <= lo:
        return None
    return float(min(1.0, max(0.0, (float(v) - lo) / (hi - lo))))


class CSMerMosReward:
    """MER + UTMOS + SECS. Same result-dict contract as the other CS rewards.

    Emits r_all / r_matrix / r_embedded (from MER), r_mos and r_spk, which
    cosyvoice/utils/grpo_kl_executor.py turns into TRAIN/r_<name> and
    TRAIN/reward_std_<name> curves without any further wiring.
    """

    # Asks grpo_kl_executor to vocode the prompt mel and hand it over as ref_wav. The SECS
    # term is the only thing that needs it, so it is requested rather than assumed -- and
    # before this flag existed the executor passed no ref_wav at all, which left w_spk
    # silently inert in every reward that accepted one.
    wants_ref_wav = False        # set per instance below: only w_spk > 0 needs it

    def __init__(self, device='cuda:0',
                 # --- content (delegated verbatim to CSMerReward) ---
                 mer_model='openai/whisper-large-v3', mer_adapter=None,
                 mer_dtype='float16', ins_weight=0.5, sharpness=3.0,
                 alpha=1.0, beta=2.0, num_beams=1, max_new_tokens=200,
                 content_scope='all',
                 # --- weights ---
                 w_content=1.0, w_mos=0.0, w_spk=0.0,
                 # --- UTMOS ---
                 mos_backend='utmos22', mos_floor=0.0, mos_lo=1.0, mos_hi=5.0,
                 utmosv2_repo='',
                 utmosv2_config='fusion_stage3', utmosv2_fold=0, mos_tmpdir='/dev/shm',
                 # --- SECS ---
                 spk_lo=0.0, spk_hi=1.0):
        self.content = CSMerReward(
            model=mer_model, adapter=mer_adapter, device=device, dtype=mer_dtype,
            alpha=alpha, beta=beta, ins_weight=ins_weight, sharpness=sharpness,
            num_beams=num_beams, max_new_tokens=max_new_tokens,
            # the content reward must stay exactly what run_grpo_mer_sft.sh trains on;
            # speaker and duration are mixed in here instead.
            w_content=1.0, w_spk=0.0, w_dur=0.0)
        if content_scope not in ('split', 'all'):
            raise ValueError(f"content_scope must be split or all: {content_scope}")
        self.content_scope = content_scope
        self.w_content, self.w_mos, self.w_spk = w_content, w_mos, w_spk
        self.mos_floor = mos_floor
        self.mos_lo, self.mos_hi = mos_lo, mos_hi
        self.spk_lo, self.spk_hi = spk_lo, spk_hi
        self.mos = None
        if w_mos > 0 or mos_floor > 0:
            if mos_backend == 'utmosv2':
                self.mos = UtmosV2(device=device, repo=utmosv2_repo,
                                   config=utmosv2_config, fold=utmosv2_fold,
                                   tmpdir=mos_tmpdir)
            elif mos_backend == 'utmos22':
                self.mos = Utmos22(device=device)
            else:
                raise ValueError(f'mos_backend must be utmos22 or utmosv2: {mos_backend}')
        self.spk = WavLMSpeakerSim(device=device) if w_spk > 0 else None
        # one hift forward per group, so do not ask for it when nothing consumes it
        self.wants_ref_wav = self.spk is not None
        self.n_no_ref = 0

    def _to16k(self, wav, sr):
        return self.content._to16k(wav, sr)

    def __call__(self, wav, sr, gt_text, lang=None, ref_wav=None, ref_sr=None):
        out = self.content(wav, sr, gt_text, lang, ref_wav=None, ref_sr=None)
        # 'all' = 1 - tanh(sharpness * MER) over the whole utterance, i.e. the reported
        # number itself. r_matrix / r_embedded are still computed and returned either way,
        # but under 'all' they are OBSERVED ONLY -- see CONTENT SCOPE in the header.
        r_content = out.get('r_all') if self.content_scope == 'all' else out.get('r_content')
        if r_content is None:
            r_content = 0.0
        g = self._to16k(wav, sr)

        mos_raw = self.mos(g) if self.mos is not None else None
        r_mos = rescale(mos_raw, self.mos_lo, self.mos_hi)

        r_spk = None
        if self.spk is not None:
            if ref_wav is None:
                self.n_no_ref += 1
            else:
                r = self._to16k(ref_wav, ref_sr or 16000)
                # id() is stable for the duration of one group: _rewards() hands the same
                # prompt tensor to every rollout of an utterance, so the reference
                # embedding is computed once per group rather than once per rollout.
                r_spk = rescale(self.spk(g, r, ref_key=id(ref_wav)),
                                self.spk_lo, self.spk_hi)

        num = den = 0.0
        for v, w in ((r_content, self.w_content), (r_mos, self.w_mos),
                     (r_spk, self.w_spk)):
            if v is not None and w > 0:
                num += w * v
                den += w
        reward = float(num / den) if den else 0.0
        # Guardrail, not a term: it can only remove reward, so there is nothing to gain by
        # driving MOS higher -- which is the point, given GT scores BELOW the SFT model.
        if self.mos_floor > 0 and mos_raw is not None and mos_raw < self.mos_floor:
            reward = 0.0

        # r_content must report what actually entered the reward, not what CSMerReward
        # would have used: under scope='all' the split value would make the logged
        # components inconsistent with the objective.
        out.update({'reward': reward, 'r_content': r_content, 'r_mos': r_mos,
                    'r_spk': r_spk, 'mos_raw': mos_raw, 'r_dur': None})
        return out


def main():
    ap = argparse.ArgumentParser(description='Offline check of the MER+UTMOS+SECS reward')
    ap.add_argument('--lst', required=True)
    ap.add_argument('--wav_dir', default=None)
    ap.add_argument('--mer_model',
                    default='')
    ap.add_argument('--mos_backend', default='utmos22', choices=['utmos22', 'utmosv2'])
    ap.add_argument('--w_content', type=float, default=1.0)
    ap.add_argument('--w_mos', type=float, default=0.3)
    ap.add_argument('--w_spk', type=float, default=0.3)
    ap.add_argument('--limit', type=int, default=20)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    rw = CSMerMosReward(device=args.device, mer_model=args.mer_model,
                        mos_backend=args.mos_backend, w_content=args.w_content,
                        w_mos=args.w_mos, w_spk=args.w_spk)
    rows = read_lst(args.lst)[:args.limit]
    acc = {k: [] for k in ('reward', 'r_content', 'r_mos', 'r_spk', 'mos_raw', 'mer')}
    for r in rows:
        p = (os.path.join(args.wav_dir, r['utt_id'] + '.wav') if args.wav_dir
             else (r['gt_wav'] or r['prompt_wav']))
        if not os.path.exists(p):
            continue
        wav, sr = torchaudio.load(p)
        ref = r.get('prompt_wav')
        rw_wav, rw_sr = (torchaudio.load(ref) if ref and os.path.exists(ref)
                         else (None, None))
        o = rw(wav, sr, r['text'], r.get('lang'), ref_wav=rw_wav, ref_sr=rw_sr)
        f = lambda v: '-' if v is None else format(v, '.3f')
        print(f"[{r['utt_id']}] R={o['reward']:.3f}  content={f(o['r_content'])} "
              f"mos={f(o['r_mos'])}(raw {f(o['mos_raw'])}) spk={f(o['r_spk'])} "
              f"| mer={f(o.get('mer'))}")
        for k in acc:
            if o.get(k) is not None:
                acc[k].append(float(o[k]))
    m = lambda v: float('nan') if not v else float(np.mean(v))
    s = lambda v: float('nan') if len(v) < 2 else float(np.std(v))
    print('\nmean  ' + '  '.join(f'{k}={m(v):.3f}' for k, v in acc.items()))
    print('std   ' + '  '.join(f'{k}={s(v):.3f}' for k, v in acc.items()))
    print('\nstd is ACROSS utterances, which is mostly sentence difficulty. The number that '
          'decides whether a term can teach anything is the WITHIN-GROUP std over rollouts '
          'of one sentence -- tensorboard reward_std_mos / reward_std_spk during training.')


if __name__ == '__main__':
    main()
