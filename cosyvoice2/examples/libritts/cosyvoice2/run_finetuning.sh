#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
#
# SFT / fine-tuning of the CosyVoice2 LLM on the SwitchLingua code-switching corpus.
#
# Difference from run.sh (from-scratch):
#   1. --checkpoint $pretrained_model_dir/llm.pt   <- starts from the released 0.5B weights
#   2. --config conf/cosyvoice2_sft.yaml           <- lr 1e-5, warmup 100, max_epoch 5
#   3. writes to exp/cosyvoice2_sft/ so it never collides with the from-scratch run
#
# Data prep (stage 0-3) is NOT repeated here -- run.sh already produced
# data/<Lang>/<split>/parquet. This script starts at stage 5.
. ./path.sh || exit 1;

# cosyvoice/utils/train_utils.py imports deepspeed unconditionally, and deepspeed's op
# builder shells out to $CUDA_HOME/bin/nvcc at IMPORT time. There is no /usr/local/cuda on
# this machine, so a hardcoded path makes every rank die before training starts with
#   FileNotFoundError: '<CUDA_HOME>/bin/nvcc'
# Unsetting it does not help -- deepspeed then raises MissingCUDAException. nvcc does live
# inside several conda envs, so point at one. Nothing here ever builds a deepspeed op, so
# the toolkit is only asked for its version string.
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

# Restricted to the languages CosyVoice2 was actually pretrained on (paper Table 3:
# Chinese 130k h / English 30k h / Japanese 4.6k h / Korean 2.2k h). Every SwitchLingua
# utterance is code-switched *with English*, so English is covered implicitly and needs
# no separate entry.
#
# Cantonese is deliberately NOT here: Table 3 lists only "Chinese", and it is unclear
# whether that 130k h includes Cantonese. Adding it also makes it 69% of the finetune
# set (7,514 of 10,854 utts), which would swamp the other three. Add it back if you
# decide the pretraining covered it.
# langs="Mandarin Japanese Korean Cantonese"

# tag that keeps this experiment's exp/, tensorboard/ and eval dirs separate
exp_tag=cs_sft_fleurs_switchlingua_part

export CUDA_VISIBLE_DEVICES="0,1,2,3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1987
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp

exp_dir=`pwd`/exp/cosyvoice2_${exp_tag}/llm/$train_engine

# combined SwitchLingua + CS-FLEURS lists, produced by run_csfleurs.sh stage 4.
# To train on SwitchLingua alone instead, point these at data/{train,dev}.data.list
# and re-enable the per-language concat that run.sh does.
train_list=data/train_combined_part.data.list
cv_list=data/dev_combined_part.data.list

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Run SFT. Fine-tuning llm from $pretrained_model_dir/llm.pt"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  # Built by run_csfleurs.sh stage 4 (SwitchLingua $langs + CS-FLEURS xtts_train).
  # Not rebuilt here: the language mix now spans two corpora with different directory
  # layouts, so run_csfleurs.sh owns it and this script just consumes the result.
  [ -s $train_list ] || { echo "$train_list missing/empty -- run run_csfleurs.sh stage 4 first"; exit 1; }
  [ -s $cv_list ] || { echo "$cv_list missing/empty -- run run_csfleurs.sh stage 4 first"; exit 1; }
  echo "train: $(wc -l < $train_list) parquet files | dev: $(wc -l < $cv_list)"
  for model in llm; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1235" \
      ../../../cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice2_sft.yaml \
      --train_data $train_list \
      --cv_data $cv_list \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --onnx_path $pretrained_model_dir \
      --model $model \
      --checkpoint $pretrained_model_dir/$model.pt \
      --model_dir `pwd`/exp/cosyvoice2_${exp_tag}/$model/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice2_${exp_tag}/$model/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
  done
fi

# average model. max_epoch is 5, so averaging 5 would include epoch 0 -- keep it to the best 3.
average_num=3
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  for model in llm; do
    decode_checkpoint=$exp_dir/${model}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python ../../../cosyvoice/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $exp_dir \
      --num ${average_num} \
      --val_best

    # average_model only writes the val-best average; keep the final epoch too so the
    # two can be compared at inference time
    last_ckpt=$(ls -v $exp_dir/epoch_*_whole.pt 2>/dev/null | tail -1)
    if [ -n "$last_ckpt" ]; then
      echo "last-epoch checkpoint is $last_ckpt -> $exp_dir/${model}_last.pt"
      cp "$last_ckpt" "$exp_dir/${model}_last.pt"
    fi
  done
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  # Assemble a loadable model dir: CosyVoice2() needs llm/flow/hift/yaml side by side,
  # but training only produces llm.pt, so link the rest from the pretrained release.
  pre_base=$(dirname $pretrained_model_dir)
  for variant in best last; do
    if [ "$variant" = "best" ]; then src=$exp_dir/llm.pt; else src=$exp_dir/llm_last.pt; fi
    [ -f "$src" ] || { echo "skip $variant: $src not found"; continue; }
    eval_dir=$pre_base/eval_${exp_tag}_$variant
    echo "build $eval_dir  (llm <- $src)"
    mkdir -p $eval_dir
    # -n so re-running does not follow an existing symlinked dir and nest inside it
    ln -sfn $pretrained_model_dir/* $eval_dir/
    # override llm.pt with the trained checkpoint (remove the symlink first).
    # strip training metadata (epoch/step) so inference can load with strict=True
    rm -f $eval_dir/llm.pt
    python -c "import torch,sys; sd=torch.load(sys.argv[1],map_location='cpu'); [sd.pop(k,None) for k in ('epoch','step')]; torch.save(sd,sys.argv[2])" "$src" "$eval_dir/llm.pt"
  done
fi
