#!/usr/bin/env bash
# Code-switching test set, all 25 languages: SwitchLingua (11) + CS-FLEURS read_test (14).
#
# Built in self-prompt / reconstruction mode -- every utterance is its own prompt, so
# prompt_text == target_text and prompt_wav IS the ground truth. utt_id is the original
# utterance id, so <save_dir>/<utt_id>.wav pairs 1:1 with the reference audio for WER/SIM.
#
#   python local/make_seedtts_lst.py --des_dir data/cs_testset_all --per_lang \
#          --extra_dirs 'data_csfleurs/read_test/*/test' --self_prompt
#
# Runs one pass per language so results land in results/<RESULT_TAG>/<Lang>/<model>/ and
# per-language scores need no demuxing. Set LANGS="" for a single combined pass.
set -euo pipefail

COSYVOICE_ROOT=""
DATA_ROOT="$COSYVOICE_ROOT/examples/libritts/cosyvoice2/data"
HERE="$(dirname "$(realpath "$0")")"

# SwitchLingua labels are capitalised language names, CS-FLEURS labels are <iso>-eng pairs.
# Keep these in sync with run_csfleurs.sh, which uses the same Option 1 / Option 2 split.

# ---- Option 1 (active): only the languages the LLM was trained on -- 2,688 utts ----
TESTSET_DIR="$DATA_ROOT/cs_testset_part"
RESULT_TAG=cs_part
# SL_LANGS=""
SL_LANGS="Mandarin Cantonese Japanese Korean"
CF_LANGS="kor-eng jpn-eng cmn-eng"
# CF_LANGS=""

# ---- Option 2: every language in both corpora -- 7,070 utts ------------------------
# TESTSET_DIR="$DATA_ROOT/cs_testset_all"
# RESULT_TAG=cs_all
# SL_LANGS="Arabic Cantonese French German Hindi Italian Japanese Korean Mandarin Russian Spanish"
# CF_LANGS="ara-eng ces-eng cmn-eng deu-eng fra-eng hin-eng ita-eng jpn-eng kor-eng por-eng rus-eng slk-eng spa-eng tel-eng"

LANGS="${LANGS-$SL_LANGS $CF_LANGS}"
# To run a subset without editing:  LANGS="Korean Mandarin cmn-eng" bash run_infer_cs.sh

# wav paths inside the lst are absolute, so PROMPT_DIR is not needed
if [[ -z "$LANGS" ]]; then
  META="$TESTSET_DIR/meta.lst"
  SAVE_BASE_DIR="examples/libritts/cosyvoice2/infer/results/${RESULT_TAG}/all"
  [[ -f "$META" ]] || { echo "[ERR] $META not found -- run local/make_seedtts_lst.py first"; exit 1; }
  echo "[INFO] combined run: $(wc -l < "$META") lines"
  source "$HERE/_infer_common.sh"
else
  total=0
  for lang in $LANGS; do
    [[ -f "$TESTSET_DIR/${lang}.lst" ]] && total=$(( total + $(wc -l < "$TESTSET_DIR/${lang}.lst") ))
  done
  echo "[INFO] $(echo $LANGS | wc -w) languages, $total utterances per model"
  for lang in $LANGS; do
    META="$TESTSET_DIR/${lang}.lst"
    SAVE_BASE_DIR="examples/libritts/cosyvoice2/infer/results/${RESULT_TAG}/${lang}"
    if [[ ! -f "$META" ]]; then
      echo "[WARN] skip $lang: $META not found"
      continue
    fi
    echo "========================================"
    echo "[INFO] lang=$lang  lines=$(wc -l < "$META")"
    source "$HERE/_infer_common.sh"
  done
fi
