#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
#
# CS-FLEURS data prep (stage 0-3) + combined train/dev lists that mix CS-FLEURS with the
# SwitchLingua data already prepared by run.sh.
#
# CS-FLEURS layout differs from SwitchLingua, hence a separate script:
#   SwitchLingua : data/<Lang>/<split>/
#   CS-FLEURS    : data_csfleurs/<subset>/<pair>/<split>/
#
# Only the xtts_train subset is used (it is the only CS-FLEURS split with training data).
# Its audio is 16 kHz, which is fine here: the speech tokenizer consumes 16 kHz natively
# and the LLM never reads the 24 kHz mel. Do not reuse it for flow/hifigan training.
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

stage=4
stop_stage=4

csfleurs_dir=""
pretrained_model_dir=""

csf_subset=xtts_train

# ---- Option 1 (active): CosyVoice2 pretrained languages only ----------------
# paper Table 3 = Chinese / English / Japanese / Korean; English is the embedded
# language of every pair, so it needs no entry of its own.
csf_pairs="cmn-eng jpn-eng kor-eng"
sl_langs="Mandarin Cantonese Japanese Korean"

# ---- Option 2: every language in both corpora -------------------------------
# CS-FLEURS has no Cantonese, so Cantonese comes from SwitchLingua only.
# csf_pairs="ara-eng ces-eng cmn-eng deu-eng fra-eng hin-eng hun-eng ita-eng jpn-eng kor-eng nld-eng pol-eng por-eng rus-eng spa-eng tur-eng"
# sl_langs="Arabic Cantonese French German Hindi Italian Japanese Korean Mandarin Russian Spanish"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "CS-FLEURS data preparation, prepare wav.scp/text/utt2spk/spk2utt (no transcoding, wavs used in place)"
  python local/prepare_csfleurs_data.py \
    --src_dir $csfleurs_dir \
    --des_dir data_csfleurs \
    --subsets $csf_subset \
    --pairs $csf_pairs \
    --min_dur 1.0 \
    --max_dur 30.0 \
    --dev_ratio 0.02
fi

# NOTE stage 2 runs onnxruntime with CUDAExecutionProvider and no device index, so it
# lands on GPU 0. Prefix the whole script with e.g. CUDA_VISIBLE_DEVICES=6 to move it.
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Extract campplus speaker embedding for CS-FLEURS"
  for pair in $csf_pairs; do
    for split in train dev; do
      [ -f data_csfleurs/$csf_subset/$pair/$split/wav.scp ] || continue
      ../../../tools/extract_embedding.py --dir data_csfleurs/$csf_subset/$pair/$split \
        --onnx_path $pretrained_model_dir/campplus.onnx
    done
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Extract discrete speech token for CS-FLEURS"
  for pair in $csf_pairs; do
    for split in train dev; do
      [ -f data_csfleurs/$csf_subset/$pair/$split/wav.scp ] || continue
      ../../../tools/extract_speech_token.py --dir data_csfleurs/$csf_subset/$pair/$split \
        --onnx_path $pretrained_model_dir/speech_tokenizer_v2.onnx
    done
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare parquet for CS-FLEURS"
  for pair in $csf_pairs; do
    for split in train dev; do
      [ -f data_csfleurs/$csf_subset/$pair/$split/wav.scp ] || continue
      mkdir -p data_csfleurs/$csf_subset/$pair/$split/parquet
      ../../../tools/make_parquet_list.py --num_utts_per_parquet 1000 \
        --num_processes 10 \
        --src_dir data_csfleurs/$csf_subset/$pair/$split \
        --des_dir data_csfleurs/$csf_subset/$pair/$split/parquet
    done
  done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Build combined SwitchLingua + CS-FLEURS train/dev lists"
  # written under new names so run.sh / run_finetuning.sh, which rebuild
  # data/{train,dev}.data.list from SwitchLingua alone, cannot clobber them
  train_list=data/train_combined_part.data.list
  dev_list=data/dev_combined_part.data.list
  rm -f $train_list $dev_list
  for lang in $sl_langs; do
    [ -f data/$lang/train/parquet/data.list ] && cat data/$lang/train/parquet/data.list >> $train_list
    [ -f data/$lang/dev/parquet/data.list ] && cat data/$lang/dev/parquet/data.list >> $dev_list
  done
  for pair in $csf_pairs; do
    [ -f data_csfleurs/$csf_subset/$pair/train/parquet/data.list ] && \
      cat data_csfleurs/$csf_subset/$pair/train/parquet/data.list >> $train_list
    [ -f data_csfleurs/$csf_subset/$pair/dev/parquet/data.list ] && \
      cat data_csfleurs/$csf_subset/$pair/dev/parquet/data.list >> $dev_list
  done
  [ -s $train_list ] || { echo "$train_list is empty -- did run.sh stage 0-3 and stage 3 above complete?"; exit 1; }
  echo "  $train_list : $(wc -l < $train_list) parquet files"
  echo "  $dev_list   : $(wc -l < $dev_list) parquet files"
  echo
  echo "Point training at these with:"
  echo "  --train_data $train_list --cv_data $dev_list"
fi
