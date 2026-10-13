#!/usr/bin/env python3
# GRPO for the CosyVoice2 LLM with the FULL objective: per-token importance ratio, PPO
# clipping, and a KL penalty against the frozen reference policy.
#
# train_grpo.py is the plain-REINFORCE variant (ratio == 1, no clip, no KL); it is kept
# alongside this one so the two can be compared. See cosyvoice/utils/grpo_kl_executor.py
# for why forward() cannot be used to score rollouts here.
#
# Same skeleton as train.py -- same yaml, same parquet data.list, same optimizer/scheduler
# helpers, same checkpoint layout -- but the objective is reinforcement learning instead of
# cross entropy:
#
#   for each text in the batch
#       sample `rollout_n` speech-token sequences from the current policy
#       decode each with the frozen flow+hift and score it
#       advantage = (reward - group mean) / group std          <- GRPO, no value network
#       loss = -mean(advantage * logP(rollout)) + kl_coef * KL(policy || reference)
#
# The reward is the language-decomposed WER from examples/grpo/cosyvoice2: the utterance is
# force-aligned against its ground-truth text, each language span is cropped and transcribed
# with ITS OWN ASR tag, and R_matrix / R_embedded are scored separately. An utterance-level
# WER cannot supply a gradient for the embedded language -- SenseVoice writes Korean-embedded
# English in Hangul, so its Latin preservation on kor-eng is 3% even on ground-truth audio.
#
# No Triton server: the reward is computed in-process, once per rank.
#
# The reference audio in each batch doubles as the flow prompt, so the rollout is scored in
# the utterance's own voice rather than a fixed one.
import argparse
import datetime
import logging
import os
import sys

import torch
import torch.distributed as dist
from hyperpyyaml import load_hyperpyyaml
from torch.distributed.elastic.multiprocessing.errors import record

from cosyvoice.utils.executor import Executor  # noqa: F401  (kept for parity with train.py)
from cosyvoice.utils.train_utils import (
    check_modify_and_save_config, init_dataset_and_dataloader, init_distributed,
    init_optimizer_and_scheduler, init_summarywriter, save_model, wrap_cuda_model)

logging.getLogger('matplotlib').setLevel(logging.WARNING)


def get_args():
    parser = argparse.ArgumentParser(description='GRPO training for the CosyVoice2 llm')
    parser.add_argument('--train_engine', default='torch_ddp', choices=['torch_ddp'],
                        help='deepspeed is not supported: rollout sampling needs the raw module')
    parser.add_argument('--model', default='llm', choices=['llm'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--cv_data', required=True)
    parser.add_argument('--qwen_pretrain_path', required=True)
    parser.add_argument('--onnx_path', required=True)
    parser.add_argument('--checkpoint', required=True,
                        help='policy init. RL fine-tunes an existing model; training the '
                             'speech head from random with a sparse reward does not work.')
    parser.add_argument('--pretrained_model_dir', required=True,
                        help='holds flow.pt / hift.pt, frozen, used to turn tokens into audio')
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--tensorboard_dir', required=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--prefetch', type=int, default=100)
    parser.add_argument('--pin_memory', action='store_true', default=False)
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--accum_grad', type=int, default=None,
                        help='override train_conf.accum_grad. Effective batch is '
                             'gpus x max_frames_in_batch x accum_grad, so halving the '
                             'batch and doubling this leaves it unchanged -- which is what '
                             'keeps a memory fix from also becoming a different experiment.')
    parser.add_argument('--max_frames_in_batch', type=int, default=None,
                        help='override the config batch size. GRPO memory is dominated by '
                             'rollouts, not by the batch itself: every item is sampled '
                             'rollout_n times and each rollout is decoded to audio, so a '
                             'batch that fits at step 0 can OOM later once the policy '
                             'starts emitting longer sequences. Halving this halves the '
                             'peak; double accum_grad to keep the effective batch.')
    parser.add_argument('--seg_norm', default=None, choices=['masked', 'all'],
                        help="override grpo_conf.seg_norm for a segment-wise run. 'masked' "
                             'averages the policy term over the masked tokens; \'all\' '
                             'divides by the full rollout length instead, which makes the '
                             "step size scale with each utterance's English fraction. The "
                             'run script appends it to exp_tag, so the two variants never '
                             'share a checkpoint directory.')
    parser.add_argument('--timeout', type=int, default=1800,
                        help='monitored_barrier timeout for the uneven-workload join. Must '
                             'exceed the slowest BATCH, not the slowest step: a GRPO batch '
                             'is ~76s (4 utts x 4 rollouts), so train.py\'s 60s default '
                             'fires every batch and silently ends the epoch after ~10.')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='resume: first epoch index to run. Epoch indices drive the data '
                             'shuffle seed (set_epoch) and checkpoint names, so resuming '
                             'after epoch 0 means --start_epoch 1, not a rerun of epoch 0 '
                             'that would also overwrite its epoch_0_step_* checkpoints.')
    parser.add_argument('--start_step', type=int, default=0,
                        help='resume: optimizer-step counter to continue from. It is the '
                             'tensorboard x axis and the clock of any step-based schedule '
                             '(Prefix-RFT prefix_low), so both continue instead of restarting.')
    parser.add_argument('--ref_checkpoint', default=None,
                        help='resume: weights for the FROZEN KL reference policy. By default '
                             'the reference is a copy of --checkpoint, which is right for a '
                             'fresh run but wrong for a resumed one: resuming from a step '
                             'checkpoint would silently move the KL anchor from the original '
                             'start point to the half-trained policy. Pass the original '
                             'init checkpoint here.')
    parser.add_argument('--ddp.dist_backend', dest='dist_backend', default='nccl')
    parser.add_argument('--deepspeed_config', default=None)
    parser.add_argument('--deepspeed.save_states', dest='save_states', default='model_only')
    parser.add_argument('--dpo', action='store_true', default=False)
    parser.add_argument('--reward_device', default=None,
                        help='where the aligner/ASR live; default: the training rank GPU')
    return parser.parse_args()


def build_reward(cfg, device):
    """Reward from examples/grpo/cosyvoice2, selected by grpo_conf.reward.

    'mer'      utterance-level reward, r = 1 - tanh(sharpness * MER) of the whole rollout,
               transcribed by the fine-tuned Whisper (conventional GRPO baseline)
    'segment'  per-segment reward, r = max(0, 1 - CER) of each segment (SegRPO). The
               segment unit is grpo_conf.seg_unit: 'switch' (language segments, SegRPO),
               'cs' (all matrix vs. all embedded), or 'bpe' (BPE pieces of the policy
               tokenizer).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.abspath(os.path.join(here, '../../examples/grpo/cosyvoice2')))
    g = cfg['grpo_conf']
    kind = g.get('reward', 'segment')
    common = dict(device=device, alpha=g.get('alpha', 1.0), beta=g.get('beta', 2.0))
    if kind == 'mer':
        # Conventional (utterance-level) GRPO: the mix error rate of the whole rollout,
        # transcribed by the CS-fine-tuned Whisper (whisper_ft/). Use with
        # mer_content_scope: all (conf/cosyvoice2_grpo_merall.yaml).
        from reward_cs_mer import CSMerReward
        return CSMerReward(model=g.get('mer_model', 'openai/whisper-large-v3'),
                           adapter=g.get('mer_adapter'),
                           dtype=g.get('mer_dtype', 'float16'),
                           ins_weight=g.get('ins_weight', 0.5),
                           sharpness=g.get('sharpness', 3.0),
                           num_beams=g.get('mer_num_beams', 1),
                           max_new_tokens=g.get('mer_max_new_tokens', 200),
                           content_scope=g.get('mer_content_scope', 'split'), **common)
    if kind == 'segment':
        # SegRPO: per-segment character error rate. The whole rollout is transcribed once,
        # the transcript is aligned to the reference, and every segment (seg_unit) is
        # scored on its own; the executor normalises each segment across the rollouts of
        # the group and puts its advantage on that segment's speech tokens.
        from reward_cs_segment import CSSegmentReward
        return CSSegmentReward(model=g.get('mer_model', 'openai/whisper-large-v3'),
                               adapter=g.get('mer_adapter'),
                               dtype=g.get('mer_dtype', 'float16'),
                               ins_weight=g.get('ins_weight', 0.5),
                               sharpness=g.get('sharpness', 3.0),
                               num_beams=g.get('mer_num_beams', 1),
                               max_new_tokens=g.get('mer_max_new_tokens', 200),
                               seg_unit=g.get('seg_unit', 'chunk'),
                               seg_chunk_chars=g.get('seg_chunk_chars', 3),
                               cross_script=g.get('seg_cross_script', 'zero'),
                               gate=g.get('seg_gate'),
                               bpe_tokenizer=g.get('seg_bpe_tokenizer'),
                               mos_backend=(g.get('seg_mos_backend', 'utmos22')
                                            if g.get('seg_w_mos', 0.0) > 0 else None),
                               mos_lo=g.get('seg_mos_lo', 1.0),
                               mos_hi=g.get('seg_mos_hi', 5.0),
                               utmosv2_repo=g.get('seg_utmosv2_repo',
                                                  ''),
                               utmosv2_config=g.get('seg_utmosv2_config', 'fusion_stage3'),
                               utmosv2_fold=g.get('seg_utmosv2_fold', 0), **common)
    raise ValueError(f'unknown grpo_conf.reward: {kind}')


class Token2Wav:
    """Frozen flow + hift. Only used to turn a rollout into audio for scoring."""

    def __init__(self, configs, pretrained_model_dir, device, fp16=True):
        self.device, self.fp16 = device, fp16
        self.flow = configs['flow']
        self.flow.load_state_dict(
            torch.load(f'{pretrained_model_dir}/flow.pt', map_location='cpu'), strict=False)
        self.flow.to(device).eval()
        self.hift = configs['hift']
        self.hift.load_state_dict(
            torch.load(f'{pretrained_model_dir}/hift.pt', map_location='cpu'), strict=False)
        self.hift.to(device).eval()
        for p in list(self.flow.parameters()) + list(self.hift.parameters()):
            p.requires_grad = False

    @torch.inference_mode()
    def __call__(self, token, prompt_token, prompt_feat, embedding):
        """token: (1, T) generated speech tokens -> 24 kHz waveform (1, N).

        Signatures verified with inspect, because the module has near-identical pairs and
        picking the wrong one throws inside the reward's except-block, which shows up as
        "every reward is 0" rather than as an error:
          flow  CausalMaskedDiffWithXvec.inference(..., streaming, finalize)
                (the non-causal MaskedDiffWithXvec is the one that takes flow_cache)
          hift  HiFTGenerator.inference(speech_feat, cache_source) -> (speech, source)
                (CausalHiFTGenerator, its subclass, is the one that takes finalize)
        conf/cosyvoice2_grpo.yaml builds HiFTGenerator, i.e. the cache_source variant.
        """
        with torch.cuda.amp.autocast(self.fp16):
            mel, _ = self.flow.inference(
                token=token.to(self.device, dtype=torch.int32),
                token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                prompt_token=prompt_token.to(self.device, dtype=torch.int32),
                prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                prompt_feat=prompt_feat.to(self.device),
                prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                embedding=embedding.to(self.device),
                streaming=False,
                finalize=True)
        speech, _ = self.hift.inference(speech_feat=mel.float(),
                                        cache_source=torch.zeros(1, 1, 0, device=self.device))
        return speech

    @torch.inference_mode()
    def prompt_wav(self, prompt_feat):
        """The PROMPT's own audio, for rewards that need a speaker reference.

        The batch carries the prompt as a mel (compute_fbank output, (1, T, 80)), never as
        a waveform, and the executor has no file path to fall back on -- cs-fleurs and
        SwitchLingua keep their audio in different trees. hift is a mel vocoder, so running
        it on prompt_feat alone reconstructs that audio without flow and without a path.
        It is a vocoder round-trip rather than the original file, which is immaterial for
        speaker identity and is applied to the reference only.

        One call per GROUP (the prompt is shared by all rollouts of an utterance), so the
        cost is amortised over rollout_n.
        """
        f = prompt_feat.to(self.device).float()
        if f.dim() == 2:
            f = f.unsqueeze(0)
        # (1, T, 80) from the dataloader; hift wants the (1, 80, T) layout flow emits
        if f.size(-1) == 80 and f.size(1) != 80:
            f = f.transpose(1, 2)
        speech, _ = self.hift.inference(
            speech_feat=f, cache_source=torch.zeros(1, 1, 0, device=self.device))
        return speech


def rollout_logp(module, text_token, text_token_len, speech_token, speech_token_len):
    """log P(speech_token | text_token) under the current policy, with gradient.

    Rebuilds exactly the sequence Qwen2LM.forward builds, so the rollout is scored by the
    same model that produced it.
    """
    batch = {
        'text_token': text_token, 'text_token_len': text_token_len,
        'speech_token': speech_token, 'speech_token_len': speech_token_len,
        'embedding': torch.zeros(text_token.size(0), 192, device=text_token.device),
    }
    out = module(batch, text_token.device)
    # forward() returns the mean CE over the target; multiply back to a sequence log-prob
    return -out['loss'] * speech_token_len.sum()


def frozen_copy(model, device):
    """A frozen copy of the policy, for the KL reference term.

    copy.deepcopy(model) raises

        TypeError: cannot pickle 'onnxruntime.capi.onnxruntime_pybind11_state.InferenceSession'

    because Qwen2LM keeps a SpeechTokenExtractor (an onnxruntime session) in its __dict__
    whenever cosyvoice.utils.onnx.online_feature is on. The session is not model state and
    the reference policy never touches it -- it only ever runs token_logps -- so unhook it
    for the duration of the copy and hand the same session (not a copy) to both models
    afterwards.
    """
    import copy
    stash = {k: model.__dict__.pop(k) for k in list(model.__dict__)
             if k.endswith('speech_token_extractor')}
    try:
        ref = copy.deepcopy(model)
    finally:
        model.__dict__.update(stash)
    ref.__dict__.update(stash)      # shared, not duplicated: an ORT session is stateless
    ref = ref.to(device).eval()
    for p in ref.parameters():
        p.requires_grad = False
    return ref


@record
def main():
    args = get_args()
    os.environ['onnx_path'] = args.onnx_path
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # flow/hift are NOT overridden to None here (unlike train.py): the frozen pair is
    # needed to decode rollouts into audio for the reward.
    override_dict = {'hifigan': None, 'qwen_pretrain_path': args.qwen_pretrain_path}
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)
    assert 'grpo_conf' in configs, 'this config has no grpo_conf block; use conf/cosyvoice2_grpo_kl.yaml'
    if args.max_frames_in_batch:
        b = configs.get('batch')
        kw = getattr(b, 'keywords', None)
        if kw is None or 'max_frames_in_batch' not in kw:
            raise SystemExit('--max_frames_in_batch: config batch has no such keyword')
        old = kw['max_frames_in_batch']
        kw['max_frames_in_batch'] = args.max_frames_in_batch
        logging.info('max_frames_in_batch %s -> %s (effective batch changed; '
                     'compensate with accum_grad if comparing against another run)',
                     old, args.max_frames_in_batch)
    # train_conf is updated from vars(args) below, so an unset override must be filled
    # with the config's own value or it would overwrite it with None
    if args.accum_grad is None:
        args.accum_grad = configs['train_conf'].get('accum_grad', 1)
    elif args.accum_grad != configs['train_conf'].get('accum_grad'):
        logging.info('accum_grad %s -> %s',
                     configs['train_conf'].get('accum_grad'), args.accum_grad)
    configs['train_conf'].update(vars(args))

    if args.train_engine == 'torch_ddp':
        # NCCL's watchdog must outlast the gloo join. At the end of an epoch the rank whose
        # shard runs out first waits in dist.barrier() (NCCL), while the others wait in
        # cosyvoice_join's monitored_barrier (gloo, --timeout = 1800 s) until they give up
        # on it. With torch's default 10 min NCCL timeout the watchdog fired first and
        # aborted the job at the end of EVERY epoch -- a run configured for 6 epochs ended
        # after one, and the run script (no set -e) went straight on to averaging.
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
        dist.init_process_group(args.dist_backend,
                                timeout=datetime.timedelta(seconds=args.timeout + 1800))
    else:
        init_distributed(args)
    _, _, train_data_loader, cv_data_loader = init_dataset_and_dataloader(args, configs, False, False)
    configs = check_modify_and_save_config(args, configs)
    writer = init_summarywriter(args)

    model = configs[args.model]
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    logging.info('policy initialised from %s', args.checkpoint)

    rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{rank}')

    g = configs['grpo_conf']
    if args.seg_norm and args.seg_norm != g.get('seg_norm'):
        logging.info('seg_norm overridden: %s -> %s', g.get('seg_norm'), args.seg_norm)
        g['seg_norm'] = args.seg_norm
    ref_model = None
    if g.get('kl_coef', 0.02) > 0:
        ref_model = frozen_copy(model, device)
        if args.ref_checkpoint:
            ref_model.load_state_dict(torch.load(args.ref_checkpoint, map_location='cpu'),
                                      strict=False)
            logging.info('reference policy for KL loaded from %s (coef %s)',
                         args.ref_checkpoint, g['kl_coef'])
        else:
            logging.info('reference policy kept for KL (coef %s)', g['kl_coef'])

    token2wav = Token2Wav(configs, args.pretrained_model_dir, device,
                          fp16=g.get('token2wav_fp16', True))
    reward_fn = build_reward(configs, args.reward_device or f'cuda:{rank}')

    model = wrap_cuda_model(args, model)
    model, optimizer, scheduler, _, _ = init_optimizer_and_scheduler(args, configs, model, False)

    if g.get('segment_advantage', False):
        # SegRPO: one advantage per segment, landing only on that segment's speech tokens
        from cosyvoice.utils.grpo_segadv_executor import GRPOSegAdvExecutor
        executor = GRPOSegAdvExecutor(g, token2wav, reward_fn, ref_model)
    else:
        # conventional GRPO: one advantage per rollout, on every token
        from cosyvoice.utils.grpo_kl_executor import GRPOKLExecutor
        executor = GRPOKLExecutor(g, token2wav, reward_fn, ref_model)
    executor.device = device

    info_dict = deepcopy_conf(configs['train_conf'])
    group_join = dist.new_group(backend='gloo',
                                timeout=datetime.timedelta(seconds=args.timeout))
    executor.step = args.start_step
    if args.start_epoch or args.start_step:
        logging.info('resuming at epoch %d, step %d (max_epoch %d)',
                     args.start_epoch, args.start_step, info_dict['max_epoch'])
    for epoch in range(args.start_epoch, info_dict['max_epoch']):
        executor.epoch = epoch
        train_data_loader.dataset.set_epoch(epoch)
        executor.train_one_epoc(model, optimizer, scheduler, train_data_loader,
                                cv_data_loader, writer, info_dict, None, group_join)
    dist.destroy_process_group(group_join)


def deepcopy_conf(conf):
    import copy
    return copy.deepcopy(dict(conf))


if __name__ == '__main__':
    main()
