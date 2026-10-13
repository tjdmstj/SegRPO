#!/bin/bash
# Mix error rate from the CS-FINE-TUNED whisper. Replacement for cal_wer_cs.sh.
#
# Same metric, different transcriber. cal_wer_cs.sh forces the matrix-language token into
# stock whisper-large-v3, and that decoder writes embedded English in the matrix script:
# on REAL HUMAN recordings it kept latin script in only 46.8% of Korean utterances and
# posted a 28.74% ground-truth MER floor. Every TTS system in this project scored between
# 14% and 26%, i.e. entirely inside the scorer's own error -- the differences being
# reported were not measurable.
#
#   scorer                       GT floor (SwitchLingua test)   English lost
#   stock whisper-large-v3               28.74%                     28.1%
#   whisper_ft/ LoRA (this one)           5.87%                      0.2%
#
# The MER computation is unchanged (whisper_ft/cs_metrics.py reproduces run_wer.py's
# punctuation table, zhconv normalisation and CJK-char/latin-word tokenisation), so the
# numbers stay on the same scale as before.
#
# Usage
#   bash cal_mer_whisper_ft.sh                      score every system found, then summarise
#   bash cal_mer_whisper_ft.sh <meta_lst> <wav_dir> score one directory (cal_wer_cs.sh shape)
#   ONLY=grpo bash cal_mer_whisper_ft.sh            only systems whose name matches
#   FORCE=1   bash cal_mer_whisper_ft.sh            rescore even if the jsonl exists
#   LIMIT=20  bash cal_mer_whisper_ft.sh            first 20 utterances only (smoke run)
#   SUBSET=kor-eng bash cal_mer_whisper_ft.sh       one test set only
#   SHARD=1/4 bash cal_mer_whisper_ft.sh            shard 1 of 4, balanced BY UTTERANCE COUNT
#
# Sharding exists because the work is very lopsided: cmn-eng alone is 13,180 of the 30,162
# utterances (44%), so splitting by test set across 4 GPUs would leave one card running
# four times as long as the others. SHARD packs whole (test set, system) jobs into n bins
# by longest-processing-time first, which lands within a few percent of even.
set -u

ROOT=""
CV=$ROOT/examples/libritts/cosyvoice2
FT=$ROOT/whisper_ft
# cosyvoice2 holds peft/zhconv and needs no CUDA_HOME (nothing here imports deepspeed)
PY=""
MODEL=${MODEL:-$FT/exp/whisper_lora/merged}     # LoRA already folded in; no peft needed
OUT=${OUT:-$FT/exp/whisper_lora/eval/tts}
GPU=${CUDA_VISIBLE_DEVICES:-0}
ONLY=${ONLY:-}
SUBSET=${SUBSET:-}
SHARD=${SHARD:-}                                 # i/n
FORCE=${FORCE:-}
LIMIT=${LIMIT:-}                                 # e.g. LIMIT=20 for a quick smoke run

[ -d "$MODEL" ] || { echo "no fine-tuned model at $MODEL -- run whisper_ft/run.sh then merge_adapter.py"; exit 1; }
mkdir -p "$OUT"

score() {   # <tag> <meta_lst> <wav_dir>
  local tag=$1 meta=$2 dir=$3
  local dst="$OUT/$tag.jsonl"
  [ -f "$meta" ] || { echo "  skip $tag: no $meta"; return; }
  [ -d "$dir"  ] || { echo "  skip $tag: no $dir";  return; }
  if [ -f "$dst" ] && [ -z "$FORCE" ]; then echo "  have $tag"; return; fi
  echo "  == $tag"
  CUDA_VISIBLE_DEVICES=$GPU $PY "$FT/infer.py" \
      --model "$MODEL" --meta_lst "$meta" --wav_dir "$dir" --out "$dst" \
      ${LIMIT:+--limit $LIMIT} \
      2>&1 | grep -E "^(lang|Cantonese|Japanese|Korean|Mandarin|MICRO|MACRO|-----)" || true
}

# ---- single-directory mode, so this drops straight into old command lines --------------
if [ $# -eq 2 ]; then
  score "$(basename "$(dirname "$2")")__$(basename "$2")" "$1" "$2"
  $PY "$FT/summarize.py" "$OUT"
  exit 0
fi

# ---- otherwise: every system this project has synthesised ------------------------------
# results/cs/<system>                      scored against cs_testset_sft/meta.lst
# results/cs_part/<subset>/<system>        scored against cs_testset_part/<subset>.lst
echo "scoring under $CV/infer/results  (model: $(basename "$MODEL"))${SHARD:+  shard $SHARD}"

JOBS=$(mktemp)
for d in "$CV"/infer/results/cs/*/; do
  [ -d "$d" ] || continue
  printf '%s\t%s\t%s\n' "cs__$(basename "$d")" "$CV/data/cs_testset_sft/meta.lst" "$d"
done >> "$JOBS"
for sub in "$CV"/infer/results/cs_part/*/; do
  name=$(basename "$sub")
  [[ "$name" == _* ]] && continue                   # _boundary, _mer are not systems
  for d in "$sub"*/; do
    [ -d "$d" ] || continue
    printf '%s\t%s\t%s\n' "${name}__$(basename "$d")" "$CV/data/cs_testset_part/${name}.lst" "$d"
  done
done >> "$JOBS"

# filter, then pack into shards by utterance count (longest job first into the emptiest bin)
PLAN=$(ONLY="$ONLY" SUBSET="$SUBSET" SHARD="$SHARD" $PY - "$JOBS" <<'EOF'
import os, sys
only, subset, shard = os.environ.get('ONLY',''), os.environ.get('SUBSET',''), os.environ.get('SHARD','')
jobs = []
for line in open(sys.argv[1], encoding='utf-8'):
    tag, meta, d = line.rstrip('\n').split('\t')
    sub, _, sysname = tag.partition('__')
    if only and only not in sysname:
        continue
    if subset and sub != subset:
        continue
    try:
        n = sum(1 for _ in open(meta, encoding='utf-8'))
    except OSError:
        continue
    jobs.append((n, tag, meta, d))
if shard:
    i, k = (int(x) for x in shard.split('/'))
    load, bins = [0] * k, [[] for _ in range(k)]
    for j in sorted(jobs, reverse=True):            # longest first
        b = load.index(min(load))
        bins[b].append(j); load[b] += j[0]
    sys.stderr.write('  shard sizes: ' + '  '.join(f'{x+1}:{l}' for x, l in enumerate(load)) + '\n')
    jobs = bins[i - 1]
sys.stderr.write(f'  this run: {len(jobs)} jobs, {sum(j[0] for j in jobs)} utterances\n')
for n, tag, meta, d in jobs:
    print(f'{tag}\t{meta}\t{d}')
EOF
)
rm -f "$JOBS"

while IFS=$'\t' read -r tag meta dir; do
  [ -n "$tag" ] || continue
  score "$tag" "$meta" "$dir"
done <<< "$PLAN"

echo
$PY "$FT/summarize.py" "$OUT" --csv "$OUT/summary.csv"
echo
echo "per-language view:   $PY $FT/summarize.py $OUT --lang Korean"
echo "per-utterance detail: $OUT/<testset>__<system>.jsonl  (ref and hyp side by side)"
