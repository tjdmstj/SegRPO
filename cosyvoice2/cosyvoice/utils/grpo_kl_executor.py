# GRPO with the full objective: per-token importance ratio, PPO clipping, and a KL
# penalty against the frozen reference policy.
#
#   J = (1/G) Σ_i (1/|o_i|) Σ_t [ min(r_it·Â_i, clip(r_it, 1±ε)·Â_i) − β·D_KL(π_θ‖π_ref) ]
#   r_it = π_θ(o_it | x, o_i<t) / π_old(o_it | x, o_i<t)
#
# Differences from grpo_executor.py, and why each matters here:
#
# 1. Per-token log-probs computed by an explicit unistream build, NOT by Qwen2LM.forward.
#    forward() routes through prepare_lm_input_target, which picks a bistream layout with
#    probability 0.5 whenever speech_len/text_len > mix_ratio[1]/mix_ratio[0] (= 3). This
#    corpus sits around 6, so half of all calls scored a rollout under a *different*
#    sequence layout than the one it was sampled with, and the same rollout got a different
#    log-prob on every call. Any ratio built on that would be noise.
#
# 2. ppo_epochs > 1 reuses each rollout. Generation dominates the cost here (~2s per
#    rollout, and it is most of a 70s batch), so reusing them is where the throughput is.
#    Once reused, the samples are off-policy and the ratio + clip are what bound the step.
#
# 3. KL uses the k3 estimator, exp(d) − d − 1 with d = ref_logp − logp, which is
#    non-negative and lower variance than a plain difference. The previous executor used
#    |logp − ref_logp|, which is not a KL at all (it was inert only because kl_coef was 0).
#    Worth enabling here: the alignment reward pays for slow over-articulation, so the
#    policy has a cheap way to drift away from natural speech.
#
# 4. Gradients are all-reduced explicitly (sync_grads). DDP hooks its reduction into
#    DistributedDataParallel.forward, which point 1 makes it impossible to call, so
#    without this the ranks never exchange anything and quietly train different models.
import logging
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from cosyvoice.utils.common import IGNORE_ID
from cosyvoice.utils.train_utils import (cosyvoice_join, log_per_save, log_per_step,
                                         save_model, update_parameter_and_lr)


def all_sum(v, device):
    """Sum a python scalar across ranks (identity when not distributed)."""
    if not (dist.is_available() and dist.is_initialized()):
        return v
    t = torch.tensor([float(v)], device=device)
    dist.all_reduce(t)
    return float(t.item())


def sync_grads(model):
    """Average gradients across ranks by hand.

    DistributedDataParallel installs its all-reduce inside DDP.forward, and the GRPO
    executors never call it -- they drive the raw module so that a rollout is scored under
    the exact unistream layout it was sampled with (see the note at the top of this file).
    Nothing else reduces anything, so without this each rank steps on its own gradients:
    4 GPUs train 4 quietly diverging models and only rank 0's is ever saved.

    Every rank must arrive here with the same parameter list, so a parameter that came out
    of backward with no gradient contributes an explicit zero instead of being skipped --
    otherwise the collectives go out of order and the job hangs.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    world = dist.get_world_size()
    if world == 1:
        return
    for prm in model.parameters():
        if not prm.requires_grad:
            continue
        if prm.grad is None:
            prm.grad = torch.zeros_like(prm)
        dist.all_reduce(prm.grad)
        prm.grad /= world


class GRPOKLExecutor:
    # Components a reward may report as 'r_<name>' in its result dict. Each one gets a
    # TRAIN/r_<name> and a TRAIN/reward_std_<name> curve, and nothing else has to change
    # when a reward grows a new term -- a reward that does not report one leaves its
    # curves empty rather than showing a fabricated zero.
    #
    #   matrix / embedded   the native and English halves of a content reward
    #   all                 the same content scored WITHOUT the language split, i.e. the
    #                       reported metric itself. Logged even when it is not what the
    #                       reward optimises, so the split objective can be watched against
    #                       the number the results table will show.
    #   dur                 length or speaking-rate agreement with the reference
    #   mos                 predicted MOS (UTMOS)
    #   spk                 speaker similarity to the prompt
    COMPONENTS = ('matrix', 'embedded', 'all', 'dur', 'mos', 'spk')
    SHORT = {'matrix': 'm', 'embedded': 'e', 'all': 'a', 'dur': 'd',
             'mos': 'o', 'spk': 's'}

    def __init__(self, grpo_conf, token2wav, reward_fn, ref_model=None):
        self.g = grpo_conf
        self.token2wav = token2wav
        self.reward_fn = reward_fn
        self.ref_model = ref_model
        self.step = 0
        self.epoch = 0
        self._n_fail = 0
        self._comp = {k: [] for k in self.COMPONENTS}
        self._gstd, self._rlen = [], []      # per-group reward std, rollout length
        # per-group std of the two COMPONENTS. audio_reference_reward.md 5 turns on this:
        # scoring the whole utterance is only defensible if the native score is flat
        # across rollouts of one sentence, because GRPO learns from whatever moves inside
        # a group. If native moves as much as English, whole-utterance scoring teaches
        # prosody imitation rather than English pronunciation, and English must be split
        # out. The proxy measurement said native moves MORE; this measures it for real.
        # Same question applies to every component a reward reports, which is why these
        # are a table rather than three named pairs: a term whose within-group std is ~0 is
        # a CONSTANT inside the group, the group-mean subtraction deletes it, and it cannot
        # move the policy however large its weight. UTMOS and speaker similarity are both
        # near their ceiling on this data, so watch reward_std_mos / reward_std_spk before
        # concluding either one did anything.
        self._gstd_c = {k: [] for k in self.COMPONENTS}
        self._grp_c = {k: [] for k in self.COMPONENTS}       # components of this group
        self._by_lang = {}
        self._parse_lang = None
        self.rank = int(os.environ.get('RANK', 0))
        self.device = torch.device('cuda:{}'.format(int(os.environ.get('LOCAL_RANK', 0))))

    # ------------------------------------------------------------------ log-probs
    def token_logps(self, module, text_token, text_token_len, speech_token):
        """-> (T+1,) log π(o_t | x, o_<t) for the speech tokens plus the trailing EOS.

        Rebuilds the unistream sequence exactly as prepare_lm_input_target's else-branch
        does, so the layout is deterministic and matches how the rollout was sampled:
            input  = [sos] [text ...] [task_id] [speech ...]
            target = [IGN] [IGN  ...] [sp_0   ] [sp_1 ...  ] [eos]
        Position t of the input predicts position t of the target, so no shift is needed.
        """
        tl = int(text_token_len.item())
        text_emb = module.llm.model.model.embed_tokens(text_token[:, :tl])
        speech_emb = module.speech_embedding(speech_token)
        sos_emb = module.llm_embedding.weight[module.sos].reshape(1, 1, -1)
        task_emb = module.llm_embedding.weight[module.task_id].reshape(1, 1, -1)
        lm_input = torch.concat([sos_emb, text_emb, task_emb, speech_emb], dim=1)
        lm_input_len = torch.tensor([lm_input.size(1)], dtype=torch.int32)

        target = torch.full((1, lm_input.size(1)), IGNORE_ID,
                            dtype=torch.long, device=self.device)
        # the task_id position predicts the first speech token, and the last speech token
        # predicts eos, so the supervised span starts at 1 + text_len
        target[0, 1 + tl:] = torch.concat(
            [speech_token[0], torch.tensor([module.eos_token], device=self.device)])

        lm_output, _ = module.llm(lm_input, lm_input_len.to(self.device))
        logits = module.llm_decoder(lm_output)
        logp = F.log_softmax(logits.float(), dim=-1)
        sel = target[0, 1 + tl:]
        return logp[0, 1 + tl:].gather(-1, sel.unsqueeze(-1)).squeeze(-1)

    # ---------------------------------------------------------------- rollout & reward
    def _sample(self, module, text_token, text_token_len, embedding):
        toks = list(self._generate(module, text_token, text_token_len, embedding))
        if not toks:
            return None
        ids = [int(t.reshape(-1)[0]) if torch.is_tensor(t) else int(t) for t in toks]
        return torch.tensor([ids], dtype=torch.int64, device=self.device)

    @torch.inference_mode()
    def _generate(self, module, text_token, text_token_len, embedding):
        empty = torch.zeros(1, 0, dtype=torch.int32, device=self.device)
        zero = torch.zeros(1, dtype=torch.int32, device=self.device)
        return list(module.inference(
            text=text_token, text_len=text_token_len,
            prompt_text=empty, prompt_text_len=zero,
            prompt_speech_token=empty, prompt_speech_token_len=zero,
            embedding=embedding,
            sampling=self.g.get('top_k', 25),
            max_token_text_ratio=self.g.get('max_token_text_ratio', 20),
            min_token_text_ratio=self.g.get('min_token_text_ratio', 2)))

    def _note_fail(self, ex):
        self._n_fail += 1
        if self._n_fail <= 3:
            logging.error('reward failed (%d): %s', self._n_fail, ex, exc_info=True)
        elif self._n_fail % 200 == 0:
            logging.warning('reward failed %d times so far: %s', self._n_fail, ex)

    def _lang_of(self, utt):
        """Language label from the utterance id.

        utt_id.split('_')[0] is 'csf' for every cs-fleurs utterance, so half the corpus
        used to reach the reward with no usable language at all.
        """
        if self._parse_lang is None:
            try:    # the reward package is put on sys.path by build_reward()
                from reward_cs_whisper_ll import parse_lang
                self._parse_lang = parse_lang
            except ImportError:
                self._parse_lang = lambda u: str(u).split('_')[0]
        return self._parse_lang(utt)

    def _to_wav(self, tokens, prompt_token, prompt_feat, embedding):
        try:
            wav = self.token2wav(tokens, prompt_token, prompt_feat, embedding)
            wav = wav.squeeze(0).float().cpu()
            return wav if wav.numel() >= 2400 else None
        except Exception as ex:
            self._note_fail(ex)
            return None

    def _rewards(self, seqs, prompt_token, prompt_feat, embedding, text, lang,
                 utt=None):
        """One reward per rollout; a rollout that will not decode scores 0.

        Scored as a batch where the reward offers it. A GRPO group is `rollout_n` takes of
        the SAME sentence, so the teacher-forced Whisper reward collapses 4 forwards into
        one padded batch -- the per-span ASR calls of reward_cs_wer.py have no equivalent.
        """
        self._grp_c = {k: [] for k in self.COMPONENTS}       # this group only
        # An audio-reference reward is keyed by UTTERANCE, not by text: its target is this
        # recording's own phonemes, and in cs-fleurs one sentence is read by up to four
        # speakers. Rewards that do not declare wants_utt keep their old signature.
        kw = {'utt': utt} if getattr(self.reward_fn, 'wants_utt', False) else {}
        # A speaker-similarity term needs the PROMPT's audio, which the batch only carries
        # as a mel. Token2Wav.prompt_wav vocodes it once per group -- the prompt is the same
        # for every rollout of one utterance -- and rewards that do not declare
        # wants_ref_wav never pay for it. Without this, ref_wav stayed None and every
        # reward's w_spk was silently inert.
        if getattr(self.reward_fn, 'wants_ref_wav', False):
            try:
                rw = self.token2wav.prompt_wav(prompt_feat).squeeze(0).float().cpu()
                kw['ref_wav'], kw['ref_sr'] = rw, 24000
            except Exception as ex:
                self._note_fail(ex)
        if not getattr(self.reward_fn, 'needs_audio', True):
            # A reward on LENGTH alone needs no audio: the flow decoder emits a fixed
            # number of mel frames per speech token, so the rollout's token count already
            # is its duration, and prompt_token is the ground-truth utterance's. Skipping
            # flow+hift here is most of the cost of a GRPO batch.
            out = []
            for s in seqs:
                r = self.reward_fn.from_tokens(int(s.size(1)), int(prompt_token.size(1)))
                self._collect(r)
                out.append(float(r['reward']))
            return out

        wavs = [self._to_wav(s, prompt_token, prompt_feat, embedding) for s in seqs]
        out = [0.0] * len(seqs)
        idx = [i for i, w in enumerate(wavs) if w is not None]
        if not idx:
            return out
        n = len(idx)
        res, many = None, getattr(self.reward_fn, 'score_many', None)
        if many is not None:
            try:
                mkw = {}
                if 'utt' in kw:
                    mkw['utts'] = [utt] * n
                for f in ('ref_wav', 'ref_sr'):
                    if f in kw:
                        mkw[f] = kw[f]
                res = many([wavs[i] for i in idx], [24000] * n, [text] * n,
                           [lang] * n, **mkw)
            except Exception as ex:
                # one bad rollout must not zero the whole group -- fall back to one by one
                self._note_fail(ex)
        if res is None:
            res = []
            for i in idx:
                try:
                    res.append(self.reward_fn(wavs[i], 24000, text, lang, **kw))
                except Exception as ex:
                    self._note_fail(ex)
                    res.append({'reward': 0.0})
        for i, r in zip(idx, res):
            self._collect(r)
            out[i] = float(r['reward'])
        return out

    def _collect(self, r):
        """Stash whichever components this reward reports. A reward that does not have a
        language decomposition (duration) returns None for those and its curves stay
        empty rather than showing a fabricated zero."""
        for name in self.COMPONENTS:
            v = r.get('r_' + name)
            if v is not None:
                self._comp[name].append(float(v))
                self._grp_c[name].append(float(v))

    def _reward(self, tokens, prompt_token, prompt_feat, embedding, text, lang,
                utt=None):
        return self._rewards([tokens], prompt_token, prompt_feat, embedding, text, lang,
                             utt)[0]

    # ------------------------------------------------------------------ distributed
    def _all_sum(self, v):
        return all_sum(v, self.device)

    def _sync_grads(self, model):
        return sync_grads(model)

    # ---------------------------------------------------------------------- main loop
    def train_one_epoc(self, model, optimizer, scheduler, train_data_loader,
                       cv_data_loader, writer, info_dict, scaler, group_join):
        lr = optimizer.param_groups[0]['lr']
        info_dict['lr'] = lr
        logging.info('Epoch %s GRPO-KL info lr %s rank %s', self.epoch, lr, self.rank)
        model.train()
        module = model.module if hasattr(model, 'module') else model
        n_roll = self.g.get('rollout_n', 4)
        ppo_epochs = self.g.get('ppo_epochs', 2)
        eps = self.g.get('clip_eps', 0.2)
        kl_coef = self.g.get('kl_coef', 0.02)
        assert n_roll >= 2, 'rollout_n must be >= 2: GRPO normalises within the group'

        model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext
        with model_context():
            pbar = tqdm(enumerate(train_data_loader),
                        desc='Epoch {} GRPO-KL'.format(self.epoch), unit='batch',
                        disable=(self.rank != 0))
            for batch_idx, batch_dict in pbar:
                info_dict.update(tag='TRAIN', step=self.step, epoch=self.epoch,
                                 batch_idx=batch_idx)
                if cosyvoice_join(group_join, info_dict):
                    break

                rewards_log, n_degenerate, n_clipped, n_tok = [], 0, 0, 0
                self._comp = {k: [] for k in self.COMPONENTS}
                self._gstd, self._rlen, self._by_lang = [], [], {}
                self._gstd_c = {k: [] for k in self.COMPONENTS}
                B = batch_dict['text_token'].size(0)

                # ---- phase 1: roll out and score once (the expensive part) ----
                groups = []
                for i in range(B):
                    tt = batch_dict['text_token'][i:i + 1].to(self.device)
                    ttl = batch_dict['text_token_len'][i:i + 1].to(self.device)
                    stl = batch_dict['speech_token_len'][i:i + 1].to(self.device)
                    emb = batch_dict['embedding'][i:i + 1].to(self.device)
                    fl = int(batch_dict['speech_feat_len'][i])
                    ntok = min(fl // 2, int(stl.item()))
                    gt_tok = batch_dict['speech_token'][i:i + 1].to(self.device)[:, :ntok]
                    gt_feat = batch_dict['speech_feat'][i:i + 1].to(self.device)[:, :2 * ntok]
                    lang = self._lang_of(batch_dict['utts'][i])

                    seqs = []
                    for _ in range(n_roll):
                        seq = self._sample(module, tt, ttl, emb)
                        if seq is not None and seq.numel() > 0:
                            seqs.append(seq)
                    if len(seqs) < 2:
                        continue
                    rews = self._rewards(seqs, gt_tok, gt_feat, emb,
                                         batch_dict['text'][i], lang,
                                         batch_dict['utts'][i])
                    rt = torch.tensor(rews, dtype=torch.float32, device=self.device)
                    rewards_log.extend(rews)
                    # the WITHIN-GROUP spread is the quantity GRPO actually consumes: the
                    # advantage is (r - mean)/std inside one group, so a batch whose groups
                    # are flat carries no gradient however high its mean reward looks
                    self._gstd.append(float(rt.std(unbiased=False)))
                    for name, buf in self._grp_c.items():
                        if len(buf) > 1:
                            self._gstd_c[name].append(
                                float(torch.tensor(buf).std(unbiased=False)))
                    self._rlen.extend(int(x.size(1)) for x in seqs)
                    self._by_lang.setdefault(lang, []).extend(rews)
                    if rt.std(unbiased=False) < 1e-6:
                        n_degenerate += 1
                        continue
                    adv = (rt - rt.mean()) / (rt.std(unbiased=False) + 1e-6)

                    # log-probs under the sampling policy, frozen as the ratio denominator
                    with torch.no_grad():
                        old = [self.token_logps(module, tt, ttl, s).detach() for s in seqs]
                        ref = ([self.token_logps(self.ref_model, tt, ttl, s).detach()
                                for s in seqs] if kl_coef > 0 and self.ref_model else None)
                    groups.append((tt, ttl, seqs, adv, old, ref))

                # Whether this batch updates at all is a decision every rank has to take
                # together: _sync_grads below is a collective, so one rank skipping the
                # update while the others run it deadlocks the job.
                if self._all_sum(len(groups)) == 0:
                    self._log(pbar, batch_idx, rewards_log, n_degenerate, 0, 0)
                    continue

                # ---- phase 2: several updates on those same rollouts ----
                for _inner in range(ppo_epochs):
                    losses, pol_terms, kl_terms, ratios = [], [], [], []
                    for tt, ttl, seqs, adv, old, ref in groups:
                        # 1/G over the group, then the mean over the batch's groups
                        w = len(seqs) * max(1, len(groups))
                        for j, (seq, a) in enumerate(zip(seqs, adv)):
                            logp = self.token_logps(module, tt, ttl, seq)
                            ratio = torch.exp(logp - old[j])
                            un = ratio * a.detach()
                            cl = ratio.clamp(1 - eps, 1 + eps) * a.detach()
                            pol = torch.min(un, cl)
                            n_clipped += int((un > cl).sum()) if a > 0 else int((cl > un).sum())
                            n_tok += logp.numel()
                            ratios.append(float(ratio.mean()))
                            obj = pol
                            if ref is not None:
                                d = ref[j] - logp
                                # k3: exp(d) - d - 1 >= 0, unbiased and low variance
                                kl = torch.exp(d) - d - 1.0
                                obj = obj - kl_coef * kl
                                kl_terms.append(float(kl.mean()))
                            # obj.mean() is the 1/|o_i| token average. len(seqs), not
                            # rollout_n: a group that lost a rollout to a failed sample
                            # must still weigh the same as a full one.
                            losses.append(-obj.mean() / w)
                            pol_terms.append(float(-pol.mean()) / w)
                    total = torch.stack(losses).sum() if losses else None
                    if total is not None:
                        total.backward()
                    # A rank whose own groups all degenerated still takes part, and
                    # contributes exactly the zero gradient it should.
                    self._sync_grads(model)
                    info_dict['loss_dict'] = self._scalars(
                        total, pol_terms, kl_terms, ratios, rewards_log,
                        n_clipped, n_tok, n_degenerate, len(groups))
                    info_dict['batch_idx'] = batch_idx
                    info_dict['step'] = self.step
                    info_dict = update_parameter_and_lr(model, optimizer, scheduler,
                                                        None, info_dict)
                    # same call the SFT executor makes, so TRAIN/* lands in tensorboard
                    # with lr and grad_norm on the same x axis as the CV/* curves
                    log_per_step(writer, info_dict)
                    self.step += 1

                self._log(pbar, batch_idx, rewards_log, n_degenerate, n_clipped, n_tok)
                sps = info_dict.get('save_per_step', -1)
                if sps > 0 and self.step % sps < ppo_epochs:
                    self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False)
                    model.train()
            dist.barrier()
        self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)

    def _scalars(self, total, pol, kls, ratios, rewards, n_clipped, n_tok, degen, n_groups):
        """Everything worth a tensorboard curve for one optimizer step.

        `loss` is what is minimised; the rest is the diagnosis. In RL the loss itself is
        nearly uninformative -- it is an advantage-weighted quantity whose scale moves with
        the advantages, so it neither trends down nor tells you the policy improved.
        `reward` and `reward_std` are what to actually watch:
          reward      going up  = the policy is getting better at the task
          reward_std  going to 0 = the rollouts inside a group agree, no advantage left,
                                   training has nothing to learn from (see degen_frac)
          clip_frac   creeping up = the updates are outrunning the trust region; lower lr
                                   or ppo_epochs
          kl          creeping up = drifting from the reference policy; raise kl_coef
        """
        m = lambda v: float(sum(v) / len(v)) if len(v) else 0.0
        out = {
            'loss': float(total) if total is not None else 0.0,
            'loss_policy': float(sum(pol)) if pol else 0.0,     # the loss without the KL term
            'kl': m(kls),                                       # per-token k3 KL vs the reference
            'reward': m(rewards),
            'reward_std': m(self._gstd),                        # spread WITHIN a group
            'ratio': m(ratios),                                 # 1.0 on the first ppo epoch
            'clip_frac': (n_clipped / n_tok) if n_tok else 0.0,
            'degen_frac': (degen / (degen + n_groups)) if (degen + n_groups) else 0.0,
            'rollout_len': m(self._rlen),
        }
        # One pair of curves per component the reward reported. reward_std_<name> is the
        # only one that says whether that term can teach anything: a term flat inside a
        # group is deleted by the group-mean subtraction whatever its weight -- which is
        # what the en+dur run looked like (r_dur pinned at 0.59 for 1216 steps while
        # rollout_len still grew 0.82x -> 1.32x).
        for name in self.COMPONENTS:
            if self._gstd_c[name]:
                out['reward_std_{}'.format(name)] = m(self._gstd_c[name])
            if self._comp[name]:
                out['r_{}'.format(name)] = m(self._comp[name])
        # native-vs-English within-group spread, the number 5 is argued from: below ~0.33
        # the native score is flat enough that whole-utterance scoring is dominated by
        # English and the split is unnecessary; near or above 1.0 it is not.
        if self._gstd_c['matrix'] and self._gstd_c['embedded']:
            gm, ge = m(self._gstd_c['matrix']), m(self._gstd_c['embedded'])
            if ge > 1e-9:
                out['std_ratio_matrix_over_embedded'] = gm / ge
        # per-language reward: the whole point of this recipe is that the languages behave
        # differently, and a rising mean can hide one of them collapsing
        for lang, v in sorted(self._by_lang.items()):
            out['R_{}'.format(lang)] = m(v)
        return out

    def _log(self, pbar, batch_idx, rewards, degen, clipped, ntok):
        if self.rank != 0:
            return
        m = lambda v: round(sum(v) / len(v), 3) if v else float('nan')
        sd = lambda v: round(float(torch.tensor(v).std(unbiased=False)), 3) if len(v) > 1 else 0.0
        # gsd_* are the WITHIN-GROUP spreads (what the advantage uses); sd_* are the
        # spreads across the whole batch, which are mostly sentence difficulty
        stats = dict(step=self.step, R=m(rewards),
                     mtx=m(self._comp['matrix']), emb=m(self._comp['embedded']),
                     sd_m=sd(self._comp['matrix']), sd_e=sd(self._comp['embedded']),
                     n=len(rewards), degen=degen,
                     clip=('{:.0%}'.format(clipped / ntok) if ntok else '-'))
        # only the components this reward actually reports, so the line stays readable
        for name in self.COMPONENTS:
            if self._gstd_c[name]:
                stats['gsd_' + self.SHORT[name]] = m(self._gstd_c[name])
        pbar.set_postfix(**stats)
        every = self.g.get('log_every', 1)
        if every and (batch_idx + 1) % every == 0:
            pbar.write('  b{:<5} '.format(batch_idx + 1)
                       + ' '.join('{}={}'.format(k, v) for k, v in stats.items()))

    @torch.inference_mode()
    def cv(self, model, cv_data_loader, writer, info_dict, on_batch_end=True):
        logging.info('Epoch %s GRPO-KL CV rank %s', self.epoch, self.rank)
        model.eval()
        module = model.module if hasattr(model, 'module') else model
        rs, n_max = [], self.g.get('cv_max_utts', 64)
        self._comp = {k: [] for k in self.COMPONENTS}
        self._by_lang = {}
        for batch_dict in cv_data_loader:
            for i in range(batch_dict['text_token'].size(0)):
                if len(rs) >= n_max:
                    break
                tt = batch_dict['text_token'][i:i + 1].to(self.device)
                ttl = batch_dict['text_token_len'][i:i + 1].to(self.device)
                stl = batch_dict['speech_token_len'][i:i + 1].to(self.device)
                emb = batch_dict['embedding'][i:i + 1].to(self.device)
                ntok = min(int(batch_dict['speech_feat_len'][i]) // 2, int(stl.item()))
                gt_tok = batch_dict['speech_token'][i:i + 1].to(self.device)[:, :ntok]
                gt_feat = batch_dict['speech_feat'][i:i + 1].to(self.device)[:, :2 * ntok]
                seq = self._sample(module, tt, ttl, emb)
                if seq is None:
                    continue
                lang = self._lang_of(batch_dict['utts'][i])
                r = self._reward(seq, gt_tok, gt_feat, emb, batch_dict['text'][i], lang,
                                 batch_dict['utts'][i])
                self._by_lang.setdefault(lang, []).append(r)
                rs.append(r)
            if len(rs) >= n_max:
                break
        mean_r = sum(rs) / len(rs) if rs else 0.0
        logging.info('Epoch %s CV mean reward %.4f over %d utts', self.epoch, mean_r, len(rs))
        info_dict.setdefault('lr', 0.0)
        m = lambda v: float(sum(v) / len(v)) if len(v) else 0.0
        # plain floats: average_model_grpo.py reads these with yaml.BaseLoader and ranks
        # checkpoints by 'loss', which is -reward here so that "lower is better" holds
        info_dict['loss_dict'] = {'loss': float(-mean_r), 'reward': float(mean_r)}
        for name in self.COMPONENTS:
            if self._comp[name]:
                info_dict['loss_dict']['r_{}'.format(name)] = m(self._comp[name])
        for lang, v in sorted(self._by_lang.items()):
            info_dict['loss_dict']['R_{}'.format(lang)] = m(v)
        info_dict.update(tag='CV', step=self.step, epoch=self.epoch)
        log_per_save(writer, info_dict)
        name = ('epoch_{}_whole'.format(self.epoch) if on_batch_end
                else 'epoch_{}_step_{}'.format(self.epoch, self.step))
        save_model(model, name, info_dict)
        model.train()
