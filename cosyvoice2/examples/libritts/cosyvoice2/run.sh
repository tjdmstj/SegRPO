#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

# deepspeed (imported unconditionally by transformers/cosyvoice) shells out to
# $CUDA_HOME/bin/nvcc at IMPORT time. There is no /usr/local/cuda on this machine, so a
# hardcoded path makes every run die before it starts with
#   RuntimeError: Failed to import transformers.modeling_utils ...
#   [Errno 2] No such file or directory: '<CUDA_HOME>/bin/nvcc'
# Unsetting it does not help -- deepspeed then raises MissingCUDAException. nvcc lives
# only inside some conda envs here, so point at one. Nothing ever builds a deepspeed op,
# so the toolkit is asked for nothing but its version string.
#
# Do NOT add $CUDA_HOME/bin to PATH: when CUDA_HOME is a conda env, its bin/ shadows the
# ACTIVE env's python and the run fails with an unrelated-looking ModuleNotFoundError.
if [ -z "${CUDA_HOME:-}" ] || [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
  for c in /usr/local/cuda "" \
           "" ""; do
    [ -x "$c/bin/nvcc" ] && { export CUDA_HOME=$c; break; }
  done
fi
[ -x "${CUDA_HOME:-/nonexistent}/bin/nvcc" ] || { echo "no nvcc found; deepspeed will fail to import"; exit 1; }

stage=8
stop_stage=8

# SwitchLingua code-switching corpus: <data_dir>/<Lang>.csv + <data_dir>/<Lang>/*.m4a
data_dir=""
# transcoded 24kHz mono wav (m4a is not readable by torchaudio's soundfile backend)
wav_dir=""
pretrained_model_dir=""

# languages to use; set e.g. `langs="Korean"` to train a Korean-English CS model only
langs="Arabic Cantonese French German Hindi Italian Japanese Korean Mandarin Russian Spanish"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation: m4a -> 24kHz wav, then wav.scp/text/utt2spk/spk2utt per split"
  python local/prepare_cs_data.py \
    --src_dir $data_dir \
    --wav_dir $wav_dir \
    --des_dir data \
    --langs $langs \
    --sample_rate 24000 \
    --num_processes 32 \
    --min_dur 1.0 \
    --max_dur 30.0 \
    --dev_ratio 0.02 --min_dev_utts 20 \
    --test_ratio 0.05 --min_test_utts 100 \
    --max_heldout_ratio 0.2
fi

# NOTE embedding/token extraction is not necessary now as we support online feature extraction, but training speed will be influenced
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Extract campplus speaker embedding, you will get spk2embedding.pt and utt2embedding.pt in data/$x dir"
  for lang in $langs; do
    for split in train dev test; do
      [ -f data/$lang/$split/wav.scp ] || continue
      ../../../tools/extract_embedding.py --dir data/$lang/$split \
        --onnx_path $pretrained_model_dir/campplus.onnx
    done
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Extract discrete speech token, you will get utt2speech_token.pt in data/$x dir"
  for lang in $langs; do
    for split in train dev test; do
      [ -f data/$lang/$split/wav.scp ] || continue
      ../../../tools/extract_speech_token.py --dir data/$lang/$split \
        --onnx_path $pretrained_model_dir/speech_tokenizer_v2.onnx
    done
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
  for lang in $langs; do
    for split in train dev test; do
      [ -f data/$lang/$split/wav.scp ] || continue
      mkdir -p data/$lang/$split/parquet
      ../../../tools/make_parquet_list.py --num_utts_per_parquet 1000 \
        --num_processes 10 \
        --src_dir data/$lang/$split \
        --des_dir data/$lang/$split/parquet
    done
  done
fi

# train llm
export CUDA_VISIBLE_DEVICES="3,4,6,7"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Run train. We only support llm traning for now"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  rm -f data/train.data.list data/dev.data.list
  for lang in $langs; do
    [ -f data/$lang/train/parquet/data.list ] && cat data/$lang/train/parquet/data.list >> data/train.data.list
    [ -f data/$lang/dev/parquet/data.list ] && cat data/$lang/dev/parquet/data.list >> data/dev.data.list
  done
  for model in llm; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      ../../../cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice2.yaml \
      --train_data data/train.data.list \
      --cv_data data/dev.data.list \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --onnx_path $pretrained_model_dir \
      --model $model \
      --model_dir `pwd`/exp/cosyvoice2/$model/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice2/$model/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
  done
fi

# average model
average_num=5
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  for model in llm; do
    decode_checkpoint=`pwd`/exp/cosyvoice2/$model/$train_engine/${model}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python ../../../cosyvoice/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path `pwd`/exp/cosyvoice2/$model/$train_engine  \
      --num ${average_num} \
      --val_best
  done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  for model in llm; do
    exp_dir=`pwd`/exp/cosyvoice2/$model/$train_engine
    # decode_checkpoint=$exp_dir/${model}.pt
    # echo "do model average and final checkpoint is $decode_checkpoint"
    # python ../../../cosyvoice/bin/average_model.py \
    #   --dst_model $decode_checkpoint \
    #   --src_path $exp_dir \
    #   --num ${average_num} \
    #   --val_best

    # also save the last-epoch checkpoint as ${model}_last.pt
    last_ckpt=$(ls -v $exp_dir/epoch_*_whole.pt | tail -1)
    echo "last-epoch checkpoint is $last_ckpt -> $exp_dir/${model}_last.pt"
    cp "$last_ckpt" "$exp_dir/${model}_last.pt"
  done
fi

# if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
#   echo "Export your model for inference speedup. Remember copy your llm or flow model to model_dir"
#   python cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
#   python cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
# fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  exp_dir=`pwd`/exp/cosyvoice2/llm/$train_engine     # cosyvoice_ttl -> cosyvoice2
  pre_base=$(dirname $pretrained_model_dir)
  for variant in best last; do
    if [ "$variant" = "best" ]; then src=$exp_dir/llm.pt; else src=$exp_dir/llm_last.pt; fi
    [ -f "$src" ] || { echo "skip $variant: $src not found"; continue; }   # 없으면 건너뜀
    eval_dir=$pre_base/eval_cs_$variant                # eval_ttl_* 덮어쓰기 방지
    echo "build $eval_dir  (llm <- $src)"
    mkdir -p $eval_dir
    ln -sfn $pretrained_model_dir/* $eval_dir/         # -sf -> -sfn
    rm -f $eval_dir/llm.pt
    python -c "import torch,sys; sd=torch.load(sys.argv[1],map_location='cpu'); [sd.pop(k,None) for k in ('epoch','step')]; torch.save(sd,sys.argv[2])" "$src" "$eval_dir/llm.pt"
  done
fi