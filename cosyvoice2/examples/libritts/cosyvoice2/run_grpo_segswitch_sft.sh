#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
#
# GRPO with the decoded-MER reward from the CS-fine-tuned Whisper, STARTING FROM THE CS
# SFT CHECKPOINT.
#
# run_grpo_mer.sh is the same reward from the released pretrain checkpoint. The pair is
# not interchangeable, because in GRPO-KL the initial checkpoint is ALSO the KL reference
# policy: that run is held near pretrain, this one is held near CS SFT. Comparing the MER
# reward against the audio-reference reward therefore has to match the start point --
# run_grpo_audioref_sft.sh starts from this same SFT checkpoint, and this file is its
# counterpart.
#
# This run is the BASELINE for the phoneme/audio-reference reward: it optimises the metric
# the paper reports, directly, so "our reward beats optimising WER itself" becomes a
# measurable claim rather than an argument. Same objective as run_grpo_kl.sh (per-token
# ratio, PPO clipping, KL to the frozen reference); only grpo_conf.reward changes.
#
# LEAKAGE. The scorer was LoRA-fine-tuned on data/<lang>/train, i.e. the SwitchLingua half
# of these very prompts, so it has memorised their reference text and is lenient on them.
# A lenient scorer flattens the within-group spread the advantage depends on, so watch
# reward_std; cs-fleurs is clean (the ASR never saw it). See the langs= note below.
#
# run_grpo.sh is the plain-REINFORCE variant: one update per rollout batch, where the
# ratio is exactly 1 and clipping is inert. Both are kept so the two can be compared;
# they write to separate exp/, tensorboard/ and eval dirs.
# Same stage layout as run_finetuning.sh -- 5 train, 6 average, 8 build eval dir -- so the
# three runs line up:
#
#   run.sh             from-scratch  (conf/cosyvoice2.yaml)
#   run_finetuning.sh  SFT           (conf/cosyvoice2_sft.yaml)
#   run_grpo.sh        GRPO REINFORCE (conf/cosyvoice2_grpo_align.yaml)
#   run_grpo_kl.sh     GRPO full      (conf/cosyvoice2_grpo_${reward_mode}.yaml)   <- this file
#
# No Triton server and no verl: the rollouts are sampled by the model itself, decoded by a
# frozen flow+hift, and scored in-process by the language-decomposed WER reward
# (examples/grpo/cosyvoice2/reward_cs_wer.py).
. ./path.sh || exit 1;

# cosyvoice/utils/train_utils.py imports deepspeed unconditionally, and deepspeed's op
# builder shells out to $CUDA_HOME/bin/nvcc at IMPORT time. There is no /usr/local/cuda on
# this machine, so the hardcoded /usr/local/cuda-13.0 makes every rank die with
#   FileNotFoundError: '/usr/local/cuda-13.0/bin/nvcc'
# before training starts. Unsetting it does not help either -- deepspeed then raises
# MissingCUDAException. nvcc does exist inside several conda envs, so point at one.
# GRPO never builds a deepspeed op (the recipe cannot use deepspeed at all), so the
# toolkit is only ever asked for its version string.
if [ -z "${CUDA_HOME:-}" ] || [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
  for c in /usr/local/cuda "" \
           "" ""; do
    [ -x "$c/bin/nvcc" ] && { export CUDA_HOME=$c; break; }
  done
fi
[ -x "${CUDA_HOME:-/nonexistent}/bin/nvcc" ] || { echo "no nvcc found; deepspeed will fail to import"; exit 1; }
echo "[INFO] CUDA_HOME=$CUDA_HOME  ($($CUDA_HOME/bin/nvcc --version | grep -oE 'release [0-9.]+'))"
# NOTE: do NOT prepend $CUDA_HOME/bin to PATH. When CUDA_HOME is a conda env (which is
# where the only nvcc on this box lives) that env's bin/ shadows the ACTIVE env's python
# and torchrun, and training dies with an unrelated-looking
#   ModuleNotFoundError: No module named 'hyperpyyaml'
# deepspeed builds the nvcc path from $CUDA_HOME directly, so PATH is not involved.

stage=5
stop_stage=8

pretrained_model_dir=""

# Names the run, and must match grpo_conf.reward in conf/cosyvoice2_grpo_${reward_mode}.yaml:
#   'whisper_ll' = teacher-forced Whisper log-likelihood of the reference text (default)
#   'align'      = per-span CTC posterior from the forced alignment
#   'wer'        = crop each language span and transcribe it with its own ASR tag
# Drives every output path, so the runs never collide.
reward_mode=${REWARD_MODE:-segswitch}   # must match grpo_conf.reward in
                                  # conf/cosyvoice2_grpo_${reward_mode}.yaml

exp_tag=cs_grpo_kl_${reward_mode}_sft
# the prompt set is identical for both rewards, so the data lists are shared
data_tag=cs_grpo

# Only the languages CosyVoice2 was pretrained on (paper Table 3: Chinese / English /
# Japanese / Korean). English is the embedded language of every pair so it needs no entry.
# RL sharpens behaviour the policy already shows sometimes; it cannot teach a language the
# base model cannot produce at all -- pretrained Cantonese MER was 49.8% against a 10.4%
# ground-truth floor, so drop Cantonese here if the reward stays flat.
# LEAKAGE WARNING. The scorer was fine-tuned on data/<lang>/train, i.e. exactly these
# SwitchLingua utterances, so it has memorised their reference text and is lenient on
# them -- which can flatten the within-group spread the advantage depends on. cs-fleurs
# is clean: the ASR never saw it. For the headline comparison against the phoneme reward,
# set langs="" and train on csf_pairs alone; both rewards must then use the same prompts.
langs="Mandarin Cantonese Japanese Korean"
csf_pairs="cmn-eng jpn-eng kor-eng"
csf_subset=xtts_train

# RL starts from the CS SFT checkpoint, which is ALSO the KL reference policy. Override
# with SFT_CKPT=<path/llm.pt> to branch off a different SFT run.
SFT_CKPT=${SFT_CKPT:-""}
init_checkpoint=$SFT_CKPT
[ -f "$init_checkpoint" ] || { echo "no SFT checkpoint at $init_checkpoint -- set SFT_CKPT"; exit 1; }

# Effective batch = gpus x max_frames_in_batch x accum_grad, and every GRPO run in this
# series is pinned to 8000 so their curves are comparable:
#
#   run_grpo_kl.sh (reference)        4 GPU x 2000 x 1 = 8000
#   run_grpo_mer.sh, 2 GPU            2 GPU x 2000 x 2 = 8000
#   this script, default 4 GPU        4 GPU x 1000 x 2 = 8000   <- needs MAX_FRAMES=1000
#   run_grpo_audioref_sft.sh as run   4 GPU x 1000 x 2 = 8000
#
# max_frames_in_batch is what to lower on OOM (it cuts the per-GPU peak); double accum_grad
# to hold the product constant. Unset, both fall through to conf/cosyvoice2_grpo_mer.yaml
# (2000 / 2), so the command with no overrides is exactly what it always was.
#
# The config default (2000/2) on FOUR GPUs would be 16000, double the reference, so
# MAX_FRAMES=1000 is not optional here:
#
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#     MAX_FRAMES=1000 ACCUM=2 bash run_grpo_mer_sft.sh
MAX_FRAMES=${MAX_FRAMES:-}
ACCUM=${ACCUM:-}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=2021                       # distinct from every other recipe
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp

exp_dir=`pwd`/exp/cosyvoice2_${exp_tag}/llm/$train_engine
train_list=data/train_${data_tag}.data.list
cv_list=data/dev_${data_tag}.data.list

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "stage 4: build the train/dev lists for the pretrained-language subset"
  rm -f $train_list $cv_list
  for lang in $langs; do
    [ -f data/$lang/train/parquet/data.list ] && cat data/$lang/train/parquet/data.list >> $train_list
    [ -f data/$lang/dev/parquet/data.list ] && cat data/$lang/dev/parquet/data.list >> $cv_list
  done
  for pair in $csf_pairs; do
    [ -f data_csfleurs/$csf_subset/$pair/train/parquet/data.list ] && \
      cat data_csfleurs/$csf_subset/$pair/train/parquet/data.list >> $train_list
    [ -f data_csfleurs/$csf_subset/$pair/dev/parquet/data.list ] && \
      cat data_csfleurs/$csf_subset/$pair/dev/parquet/data.list >> $cv_list
  done
  [ -s $train_list ] || { echo "$train_list is empty -- run run.sh / run_csfleurs.sh stage 0-3 first"; exit 1; }
  echo "  $train_list : $(wc -l < $train_list) parquet files"
  echo "  $cv_list    : $(wc -l < $cv_list)"
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "stage 5: GRPO train from SFT  (reward=${reward_mode}, ${num_gpus} GPU)"
  echo "         init: $init_checkpoint"
  echo "         exp : $exp_dir"
  echo "         batch: max_frames=${MAX_FRAMES:-conf(2000)}  accum=${ACCUM:-conf(2)}  effective=${num_gpus} x ${MAX_FRAMES:-2000} x ${ACCUM:-2}"
  [ -s $train_list ] || { echo "$train_list missing/empty -- run stage 4"; exit 1; }
  # deepspeed is not supported here: rollout sampling calls model.inference() on the raw
  # module, which the deepspeed engine does not expose.
  torchrun --nnodes=1 --nproc_per_node=$num_gpus \
      --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1269" \
    ../../../cosyvoice/bin/train_grpo_kl.py \
    --train_engine $train_engine \
    --config conf/cosyvoice2_grpo_${reward_mode}.yaml \
    --train_data $train_list \
    --cv_data $cv_list \
    --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
    --onnx_path $pretrained_model_dir \
    --pretrained_model_dir $pretrained_model_dir \
    --model llm \
    --checkpoint $init_checkpoint \
    --model_dir $exp_dir \
    --tensorboard_dir `pwd`/tensorboard/cosyvoice2_${exp_tag}/llm/$train_engine \
    --ddp.dist_backend $dist_backend \
    --num_workers ${num_workers} \
    --prefetch ${prefetch} \
    --pin_memory \
    ${MAX_FRAMES:+--max_frames_in_batch $MAX_FRAMES} \
    ${ACCUM:+--accum_grad $ACCUM}
fi

# average model. RL runs are short, so average fewer checkpoints than the SFT recipe.
average_num=3
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  for model in llm; do
    decode_checkpoint=$exp_dir/${model}.pt
    echo "stage 6: average -> $decode_checkpoint"
    # val_best ranks by 'loss', which the GRPO executor stores as -mean_reward, so the
    # best-reward epochs are the ones picked.
    # average_model.py rebuilds paths as epoch_{N}_whole.pt and ignores the step, so it
    # cannot see the epoch_{N}_step_{M}.pt files that save_per_step writes.
    python ../../../cosyvoice/bin/average_model_grpo.py \
      --dst_model $decode_checkpoint \
      --src_path $exp_dir \
      --num ${average_num}

    last_ckpt=$(ls -v $exp_dir/epoch_*_step_*.pt $exp_dir/epoch_*_whole.pt 2>/dev/null | tail -1)
    if [ -n "$last_ckpt" ]; then
      echo "last-epoch checkpoint is $last_ckpt -> $exp_dir/${model}_last.pt"
      cp "$last_ckpt" "$exp_dir/${model}_last.pt"
    fi
  done
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "stage 8: build eval dirs"
  pre_base=$(dirname $pretrained_model_dir)
  for variant in best last; do
    if [ "$variant" = "best" ]; then src=$exp_dir/llm.pt; else src=$exp_dir/llm_last.pt; fi
    [ -f "$src" ] || { echo "skip $variant: $src not found"; continue; }
    eval_dir=$pre_base/eval_${exp_tag}_$variant
    echo "build $eval_dir  (llm <- $src)"
    mkdir -p $eval_dir
    ln -sfn $pretrained_model_dir/* $eval_dir/
    rm -f $eval_dir/llm.pt
    python -c "import torch,sys; sd=torch.load(sys.argv[1],map_location='cpu'); [sd.pop(k,None) for k in ('epoch','step')]; torch.save(sd,sys.argv[2])" "$src" "$eval_dir/llm.pt"
  done
fi
