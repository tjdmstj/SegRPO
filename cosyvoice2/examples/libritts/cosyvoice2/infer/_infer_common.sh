#!/usr/bin/env bash
# Shared CosyVoice2 inference runner.
# Per-testset scripts set META / SAVE_BASE_DIR / (optional) PROMPT_DIR, then source this.
set -euo pipefail

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

# export CUDA_VISIBLE_DEVICES="3"
COSYVOICE_ROOT=""
cd "$COSYVOICE_ROOT"
export PYTHONPATH="$COSYVOICE_ROOT:$COSYVOICE_ROOT/third_party/Matcha-TTS:${PYTHONPATH:-}"

# 평가할 모델들 (여기 한 곳만 고치면 모든 testset에 반영됨). stage 8에서 만든 eval dir.
MODELS=(
  # "cosyvoice2_cs_grpo_kl_whisperll_ft_sft_best:pretrained_models/eval_cs_grpo_kl_whisperll_ft_sft_best"
  
  # "cosyvoice2_cs_dpo_mer_utmos_cmi_sft_best:pretrained_models/eval_cs_dpo_mer_utmos_cmi_sft_best"
  
  ## w2v fine-tuning
  # "cosyvoice2_cs_grpo_kl_audioref_w2vft_best:pretrained_models/eval_cs_grpo_kl_audioref_w2vft_best"
  # "cosyvoice2_cs_grpo_kl_audioref_w2vft_sft_best:pretrained_models/eval_cs_grpo_kl_audioref_w2vft_sft_best"

  ## english에만 reward -> 다른 언어 masking
  # "cosyvoice2_cs_grpo_kl_mer_enspan_all_sft_best:pretrained_models/eval_cs_grpo_kl_mer_enspan_all_sft_best"
  # "cosyvoice2_cs_grpo_kl_mer_enspan_sft_best:pretrained_models/eval_cs_grpo_kl_mer_enspan_sft_best"

  ## mer reward (csf까지 fine-tuning시킨 모델로)
  # "cosyvoice2_cs_grpo_kl_merall_csf_sft_best:pretrained_models/eval_cs_grpo_kl_merall_csf_sft_best"

  ## segment-level reward
  # "cosyvoice2_cs_grpo_kl_segment_sft_best:pretrained_models/eval_cs_grpo_kl_segment_sft_best"

  ## segment-level reward (csf까지 fine-tuning시킨 모델로)
  # "cosyvoice2_cs_grpo_kl_segment_csf_sft_best:pretrained_models/eval_cs_grpo_kl_segment_csf_sft_best" 
  # "cosyvoice2_cs_grpo_kl_segswitch_sft_best:pretrained_models/eval_cs_grpo_kl_segswitch_sft_best" 
  # "cosyvoice2_cs_grpo_kl_segcs_sft_best:pretrained_models/eval_cs_grpo_kl_segcs_sft_best"
  # "cosyvoice2_cs_grpo_kl_segbpe_sft_best:pretrained_models/eval_cs_grpo_kl_segbpe_sft_best"
  "cosyvoice2_cs_grpo_kl_segswitch_mosv2_sft_best:pretrained_models/eval_cs_grpo_kl_segswitch_mosv2_sft_best"


  # segment-level reward + MOS reward
  # "cosyvoice2_cs_grpo_kl_segment_mos_sft_best:pretrained_models/eval_cs_grpo_kl_segment_mos_sft_best"
  # "cosyvoice2_cs_grpo_kl_segment_mosv2_sft_best:pretrained_models/eval_cs_grpo_kl_segment_mosv2_sft_best"

  # "cosyvoice2_cs_grpo_kl_segment_mosv2_sft_top1:pretrained_models/eval_cs_grpo_kl_segment_mosv2_sft_top1"

  # "cosyvoice2_cs_dpo_mer_utmos_cmi_mosv2_sft_best:pretrained_models/eval_cs_dpo_mer_utmos_cmi_mosv2_sft_best"
)
PROMPT_DIR="${PROMPT_DIR:-}"

for entry in "${MODELS[@]}"; do
  model_name="${entry%%:*}"
  model_dir="${entry#*:}"
  SAVE_DIR="${SAVE_BASE_DIR}/${model_name}"

  echo "[INFO] model=$model_name  meta=$META  save=$SAVE_DIR"
  cmd=(python3 examples/libritts/cosyvoice2/infer/infer_cosyvoice_meta.py
    --model_dir "$model_dir"
    --meta_file "$META"
    --save_dir "$SAVE_DIR"
  )
  if [[ -n "$PROMPT_DIR" ]]; then
    cmd+=( --prompt_dir "$PROMPT_DIR" )
  fi
  "${cmd[@]}"
  echo "[DONE] $model_name"
  echo "----------------------------------------"
done
