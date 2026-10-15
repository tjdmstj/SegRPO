# SegRPO
<a href="https://ttl-cell.github.io/SegRPO/"><img src="https://img.shields.io/badge/Demo-Page-lightgrey" alt="version"></a>
![SegRPO pipeline](assets/pipeline.png)

## Training

```bash
cd cosyvoice2/examples/libritts/cosyvoice2
MAX_FRAMES=1000 ACCUM=2 CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_grpo_segswitch_sft.sh
```
