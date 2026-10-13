#!/bin/bash
# LoRA fine-tune of whisper-large-v3 on SwitchLingua code-switched speech.
#
# Why: seed-tts-eval/cal_wer_cs.sh scores TTS by forcing the matrix-language token into
# stock whisper-large-v3. That decoder writes embedded English in the matrix script, so on
# REAL HUMAN recordings it keeps latin script in only 46.8% of Korean utterances and posts
# a 29.48% ground-truth MER floor. Nothing about a TTS system can be measured through a
# scorer carrying that much of its own error. This recipe removes it.
#
# Splits ship with the corpus and are disjoint by recording session:
#   train 10,854 utts 20.45h | dev 240 utts 0.44h | test 709 utts 1.42h
# `test` is the audio the TTS evaluation set is drawn from -- all 547 targets and all 147
# prompts of data/cs_testset_sft/meta.lst -- and train/dev touch none of it, so the
# fine-tuned model can score the TTS eval set with no leakage.
#
# large-v3 specifically: it is the first checkpoint with <|yue|>. On v2 Cantonese has to
# borrow <|zh|>, and Cantonese is 69% of this corpus.
set -e

ROOT=""
HERE=$ROOT/whisper_ft
DATA=$ROOT/examples/libritts/cosyvoice2/data
PY=""      # peft 0.20 lives here
TORCHRUN=$(dirname $PY)/torchrun                          # not on PATH unless the env is activated
# lora = ~15.7M trainable, 60 MB adapter, base frozen.
# full = all 1.55B trained; needs gradient checkpointing and lr 1e-5, and carries a real
#        forgetting risk -- 20.5h of four languages can damage the other 95 the checkpoint
#        knows. Fine if this model is only ever the CS scorer.
METHOD=${METHOD:-lora}
EXP=$HERE/exp/whisper_$METHOD

# GPUs 0-3 were busy when this was written; 4-7 were free. 48 GB each, so large-v3 in
# bf16 with LoRA needs no 8-bit quantisation and no bitsandbytes.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NGPU=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')

# overridable:  stage=1 stop_stage=1 bash run.sh
# Language temperature. The corpus is 69% Cantonese / 7.6% Korean, and on this exact
# corpus a 69/7.6 split is what let the TTS SFT improve the dominant language while each
# minority one degraded in its own direction, invisibly, behind an averaged metric.
# 0.5 upsamples the minorities (Korean x3.0) without discarding any Cantonese.
# Set 1.0 for natural proportions.
BALANCE_ALPHA=${BALANCE_ALPHA:-0.5}

stage=${stage:-0}
stop_stage=${stop_stage:-3}

cd $HERE

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  echo "== stage 0: data + leakage check"
  $PY - <<'EOF'
import sys, os
sys.path.insert(0, '.')
from cs_data import read_split
DATA = ''
L = ['Cantonese', 'Mandarin', 'Japanese', 'Korean']
sets = {}
for s in ('train', 'dev', 'test'):
    rows = read_split(DATA, L, s)
    sets[s] = {r['utt_id'] for r in rows}
    per = {l: sum(1 for r in rows if r['lang'] == l) for l in L}
    print(f'  {s:<6} {len(rows):>6} utts   ' + '  '.join(f'{l[:3]}={n}' for l, n in per.items()))
for a, b in (('train', 'dev'), ('train', 'test'), ('dev', 'test')):
    n = len(sets[a] & sets[b])
    print(f'  {a} n {b} = {n}' + ('' if n == 0 else '   <-- LEAKAGE'))
    assert n == 0
meta = os.path.join(DATA, 'cs_testset_sft', 'meta.lst')
if os.path.exists(meta):
    ids = set()
    for line in open(meta, encoding='utf-8'):
        u = line.split('|')[0]
        ids.update(u.split('-'))
    print(f'  TTS eval utts inside ASR train: {len(ids & sets["train"])} '
          f'| inside ASR test: {len(ids & sets["test"])}')
    assert not (ids & sets['train']), 'TTS eval audio is in the ASR training set'
print('  OK: no leakage')
EOF
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo
  echo "== stage 1: BASELINE -- stock whisper-large-v3 on the test split"
  echo "   this is the number to beat; expect MER around 30% and heavy English loss"
  $PY infer.py --split test --out $HERE/exp/whisper_lora/eval/baseline.test.jsonl
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo
  echo "== stage 2: fine-tune (method=$METHOD)"
  if [ "$NGPU" -gt 1 ]; then
    $TORCHRUN --nnodes=1 --nproc_per_node=$NGPU --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:1251 \
      train.py --method $METHOD --output_dir $EXP --balance_alpha $BALANCE_ALPHA
  else
    $PY train.py --method $METHOD --output_dir $EXP --balance_alpha $BALANCE_ALPHA
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo
  echo "== stage 3: evaluate the adapter"
  if [ "$METHOD" = "lora" ]; then
    CKPT=$EXP/adapter_best;  FLAG=--adapter
  else
    CKPT=$EXP/model_best;    FLAG=--model
  fi
  [ -d "$CKPT" ] || { echo "nothing at $CKPT -- run stage 2"; exit 1; }
  $PY infer.py --split dev  $FLAG $CKPT --out $EXP/eval/$METHOD.dev.jsonl
  $PY infer.py --split test $FLAG $CKPT --out $EXP/eval/$METHOD.test.jsonl
  echo
  echo "compare exp/whisper_lora/eval/baseline.test.jsonl against $METHOD.test.jsonl."
  echo "The two numbers that matter: MER (the scorer's own error floor) and EN lost"
  echo "(embedded English written in the matrix script)."
fi

# Once the adapter beats the baseline, point the TTS evaluation at it:
#   $PY infer.py --meta_lst $DATA/cs_testset_sft/meta.lst \
#       --wav_dir $ROOT/examples/libritts/cosyvoice2/infer/results/cs/ground_truth \
#       --adapter $EXP/adapter_best
# Run that on ground_truth first: it is the topline, and its MER is the floor every TTS
# number should be read against.
