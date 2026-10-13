# Segment-wise GRPO: one advantage per SEGMENT, not one per rollout.
#
# GRPOKLExecutor normalises the reward across a group and gives every token of a take the
# same number. grpo_segment_executor.py narrowed that to the English tokens, still with one
# number for the whole English stretch. This goes one step further: each segment of the
# sentence is scored on its own, normalised across the group on its own, and its advantage
# lands only on its own tokens. Inside ONE rollout different segments can therefore pull in
# opposite directions:
#
#   tokens:     [저는] [Netflix] [original] [seri] [를 봐요]
#   advantage:    0     -1.73      +0.67     +0.39     0
#
# WHICH SEGMENTS CARRY SIGNAL IS DECIDED BY THE DATA, not by language. A segment where the
# four takes came out the same has zero spread, so (x - mean) is zero and it contributes
# nothing -- there is nothing left to learn there. A segment where they differ does. English
# is where disagreement concentrates in this corpus, but nothing here says 'English', and a
# matrix-language segment the takes render differently is trained on exactly the same way.
#
# That property is what removes the degeneracy the whole-utterance reward suffers from.
# mer_phoneme_study measured the decoded-MER reward flat in 34.6-36.2% of groups: every take
# scored identically and the group was dropped. Here the unit of degeneracy is a segment, so
# one hopeless word no longer costs the sentence its gradient.
#
# A rollout whose every segment is flat is DROPPED from the group rather than trained with
# an all-zero advantage, because an all-zero take is not a neutral update -- it still counts
# in the 1/G average and quietly shrinks the step. A group with fewer than two survivors
# goes the way any degenerate group goes.
#
# grpo_conf:
#   segment_advantage: true
#   reward: segment
import logging
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from cosyvoice.utils.grpo_kl_executor import GRPOKLExecutor
from cosyvoice.utils.train_utils import (cosyvoice_join, log_per_step,
                                         update_parameter_and_lr)

TOKEN_HZ = 25.0


class GRPOSegAdvExecutor(GRPOKLExecutor):

    def __init__(self, grpo_conf, token2wav, reward_fn, ref_model=None):
        super().__init__(grpo_conf, token2wav, reward_fn, ref_model)
        g = grpo_conf
        self.seg_norm = g.get('seg_norm', 'masked')
        self.seg_kl_mask = g.get('seg_kl_mask', False)
        self.seg_std_floor = g.get('seg_std_floor', 1e-6)
        self.seg_min_takes = g.get('seg_min_takes', 2)
        # A SECOND advantage, from a whole-utterance naturalness score, added on top of the
        # per-segment one. It is deliberately not folded into the segment scores: MOS is a
        # property of the whole take, so it gets the whole take's tokens, including the
        # matrix-language ones that the segment term leaves at zero. 0 = off, and then the
        # reward loads no MOS model at all.
        self.w_mos = g.get('seg_w_mos', 0.0)
        if self.seg_norm not in ('masked', 'all'):
            raise ValueError("seg_norm must be 'masked' or 'all'")
        if not getattr(reward_fn, 'wants_segments', False):
            raise ValueError('segment_advantage needs a reward that reports seg_scores and '
                             'seg_spans (reward: segment)')
        self._seg = {}

    # ---------------------------------------------------------------- reward + segments
    def _rewards(self, seqs, prompt_token, prompt_feat, embedding, text, lang, utt=None):
        """The parent's scalars, with each rollout's segment scores and spans kept aside.

        _collect is the one hook that still sees the reward dicts, so the per-segment data is
        stashed there; a rollout whose audio failed to vocode never reaches it and is padded
        back by position so the lists stay aligned with `seqs`.
        """
        self._seg_buf = []
        out = super()._rewards(seqs, prompt_token, prompt_feat, embedding, text, lang, utt)
        buf = list(self._seg_buf)
        buf += [{}] * max(0, len(seqs) - len(buf))
        self._last_seg = buf[:len(seqs)]
        return out

    def _collect(self, r):
        super()._collect(r)
        if hasattr(self, '_seg_buf'):
            # hyp is not used by the loss; it is kept so demo_segment_reward.py and any
            # failure dump can show WHAT was recognised next to the score it produced
            self._seg_buf.append({'scores': r.get('seg_scores'), 'spans': r.get('seg_spans'),
                                  'kind': r.get('seg_kind'), 'hyp': r.get('hyp'),
                                  'r_mos': r.get('r_mos')})

    # ------------------------------------------------------------------ advantages
    def seg_advantages(self, seqs, seg):
        """-> (list of (T+1,) advantage vectors or None, stats).

        Normalisation is per SEGMENT across the takes of the group, which is the whole point:
        a segment is compared only with the same segment of its sibling rollouts.
        """
        n_roll = len(seqs)
        lens = [len(s['scores']) for s in seg if s.get('scores') is not None]
        if not lens:
            return [None] * n_roll, {}
        n_seg = max(lens)
        S = np.full((n_roll, n_seg), np.nan, dtype=np.float64)
        for j, d in enumerate(seg):
            sc = d.get('scores')
            if sc is None:
                continue
            for k, v in enumerate(sc[:n_seg]):
                if v is not None:
                    S[j, k] = float(v)

        A = [torch.zeros(int(s.size(1)) + 1, dtype=torch.float32, device=self.device)
             for s in seqs]
        n_active, n_flat, n_act_en, n_act_all = 0, 0, 0, 0
        kinds = next((s['kind'] for s in seg if s.get('kind')), None)
        absa = []
        for k in range(n_seg):
            col = S[:, k]
            ok = ~np.isnan(col)
            if ok.sum() < self.seg_min_takes:
                continue
            mu, sd = col[ok].mean(), col[ok].std()
            if sd < self.seg_std_floor:                 # the takes agree: nothing to learn
                n_flat += 1
                continue
            n_active += 1
            if kinds is not None and k < len(kinds):
                n_act_all += 1
                n_act_en += int(kinds[k] == 'en')
            for j in range(n_roll):
                if not ok[j]:
                    continue
                spj = seg[j].get('spans')
                sp = spj[k] if (spj and k < len(spj)) else None
                if sp is None:
                    continue
                T = int(seqs[j].size(1))
                i0 = max(0, min(int(np.floor(sp[0] * TOKEN_HZ)), T))
                i1 = max(0, min(int(np.ceil(sp[1] * TOKEN_HZ)), T))
                if i1 <= i0:
                    continue
                a = float((col[j] - mu) / (sd + 1e-8))
                A[j][i0:i1] = a
                absa.append(abs(a))
        # the whole-utterance term: one advantage per take, spread over ALL its tokens.
        # Normalised inside the group like everything else, so it is the RANKING of the four
        # takes' naturalness that matters, not the absolute MOS.
        mos_abs = float('nan')
        if self.w_mos > 0:
            mv = np.array([s.get('r_mos') if s.get('r_mos') is not None else np.nan
                           for s in seg], dtype=np.float64)
            ok = ~np.isnan(mv)
            if ok.sum() >= self.seg_min_takes and mv[ok].std() >= self.seg_std_floor:
                mu, sd = mv[ok].mean(), mv[ok].std()
                mos_abs = 0.0
                for j in range(n_roll):
                    if ok[j]:
                        a = self.w_mos * float((mv[j] - mu) / (sd + 1e-8))
                        A[j] += a
                        mos_abs += abs(a) / max(1, int(ok.sum()))

        out = [a if float(a.abs().sum()) > 0 else None for a in A]
        stats = {'seg_total': n_seg, 'seg_active': n_active, 'seg_flat': n_flat,
                 'seg_active_en_share': (n_act_en / n_act_all) if n_act_all else float('nan'),
                 'adv_abs': (sum(absa) / len(absa)) if absa else float('nan')}
        if self.w_mos > 0:
            stats['adv_abs_mos'] = mos_abs
        return out, stats

    # ---------------------------------------------------------------------- main loop
    def train_one_epoc(self, model, optimizer, scheduler, train_data_loader,
                       cv_data_loader, writer, info_dict, scaler, group_join):
        lr = optimizer.param_groups[0]['lr']
        info_dict['lr'] = lr
        logging.info('Epoch %s GRPO-segadv info lr %s rank %s  (norm=%s, kl_mask=%s)',
                     self.epoch, lr, self.rank, self.seg_norm, self.seg_kl_mask)
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
                        desc='Epoch {} GRPO-segadv'.format(self.epoch), unit='batch',
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
                agg, cov = {}, []
                B = batch_dict['text_token'].size(0)

                # ---- phase 1: roll out, score every segment, normalise per segment ----
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
                    rewards_log.extend(rews)
                    self._by_lang.setdefault(lang, []).extend(rews)
                    self._rlen.extend(int(x.size(1)) for x in seqs)
                    rt = torch.tensor(rews, dtype=torch.float32, device=self.device)
                    self._gstd.append(float(rt.std(unbiased=False)))
                    for name, buf in self._grp_c.items():
                        if len(buf) > 1:
                            self._gstd_c[name].append(
                                float(torch.tensor(buf).std(unbiased=False)))

                    adv, st = self.seg_advantages(seqs, self._last_seg)
                    for k, v in st.items():
                        if v == v:                                  # skip NaN
                            agg.setdefault(k, []).append(v)
                    keep = [j for j, a in enumerate(adv) if a is not None]
                    if len(keep) < 2:
                        # every segment flat across the takes: the parent's degenerate case,
                        # now decided segment by segment instead of on one scalar
                        n_degenerate += 1
                        continue
                    seqs = [seqs[j] for j in keep]
                    adv = [adv[j] for j in keep]
                    cov.extend(float((a[:-1] != 0).float().mean()) for a in adv)

                    with torch.no_grad():
                        old = [self.token_logps(module, tt, ttl, s).detach() for s in seqs]
                        ref = ([self.token_logps(self.ref_model, tt, ttl, s).detach()
                                for s in seqs] if kl_coef > 0 and self.ref_model else None)
                    groups.append({'tt': tt, 'ttl': ttl, 'seqs': seqs, 'adv': adv,
                                   'old': old, 'ref': ref})

                self._seg = {k: sum(v) / len(v) for k, v in agg.items() if v}
                self._seg['adv_coverage'] = (sum(cov) / len(cov)) if cov else float('nan')

                if self._all_sum(len(groups)) == 0:
                    self._log(pbar, batch_idx, rewards_log, n_degenerate, 0, 0)
                    continue

                # ---- phase 2: PPO-clip updates, advantage differing token by token ----
                for _inner in range(ppo_epochs):
                    losses, pol_terms, kl_terms, ratios = [], [], [], []
                    for grp in groups:
                        seqs, tt, ttl = grp['seqs'], grp['tt'], grp['ttl']
                        w = len(seqs) * max(1, len(groups))
                        for j, seq in enumerate(seqs):
                            a_tok = grp['adv'][j]                   # <- a VECTOR, not a scalar
                            m = (a_tok != 0).float()
                            logp = self.token_logps(module, tt, ttl, seq)
                            ratio = torch.exp(logp - grp['old'][j])
                            un = ratio * a_tok
                            cl = ratio.clamp(1 - eps, 1 + eps) * a_tok
                            pol = torch.min(un, cl)
                            # clipping is counted where it actually bites, and the side it
                            # bites on depends on the sign of THAT token's advantage
                            n_clipped += int(((un > cl) & (a_tok > 0)).sum())
                            n_clipped += int(((cl > un) & (a_tok < 0)).sum())
                            n_tok += int(m.sum())
                            ratios.append(float((ratio * m).sum() / m.sum().clamp(min=1)))
                            denom = m.sum().clamp(min=1) if self.seg_norm == 'masked' \
                                else torch.tensor(float(m.numel()), device=self.device)
                            obj_pol = pol.sum() / denom
                            obj = obj_pol
                            if grp['ref'] is not None:
                                d = grp['ref'][j] - logp
                                kl = torch.exp(d) - d - 1.0         # k3, >= 0
                                kl = (kl * m).sum() / denom if self.seg_kl_mask else kl.mean()
                                obj = obj - kl_coef * kl
                                kl_terms.append(float(kl))
                            losses.append(-obj / w)
                            pol_terms.append(float(-obj_pol) / w)
                    total = torch.stack(losses).sum() if losses else None
                    if total is not None:
                        total.backward()
                    self._sync_grads(model)
                    info_dict['loss_dict'] = self._scalars(
                        total, pol_terms, kl_terms, ratios, rewards_log,
                        n_clipped, n_tok, n_degenerate, len(groups))
                    info_dict['batch_idx'] = batch_idx
                    info_dict['step'] = self.step
                    info_dict = update_parameter_and_lr(model, optimizer, scheduler,
                                                        None, info_dict)
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
        """The parent's curves plus the ones that say whether segment-wise credit is working.

          seg_active / seg_total   segments carrying signal. This is the headline: the
                                   fraction is what the whole design buys over one scalar
                                   per rollout. Falling towards 0 means the takes have
                                   stopped disagreeing anywhere -- either converged or
                                   sampling too cold.
          seg_flat                 segments the takes agreed on, deliberately untrained.
          seg_active_en_share      of the active segments, the share that are English. Near
                                   1 means the signal really is concentrated on the embedded
                                   language; falling means the matrix language is where the
                                   takes now differ, which is worth knowing before r_matrix
                                   drops.
          adv_coverage             fraction of a take's tokens that receive any advantage.
          adv_abs                  mean |advantage| actually applied, i.e. the signal size.
        """
        out = super()._scalars(total, pol, kls, ratios, rewards, n_clipped, n_tok,
                               degen, n_groups)
        out.update(self._seg)
        return out
