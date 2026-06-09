# TMM-SCC 实验框架

基于 TMM 的多模态对抗攻击 + 语义一致约束（SCC）扩展。目录已重构，便于上云批量实验。

## 目录结构

```
TMM/
├── data/                  # Flickr30K 图像 + SNLI-VE 标注（子集也放这里）
├── checkpoints/           # VLR/VE 模型权重
│   ├── VLR/               # albef, tcl, clip, blip
│   └── VE/                # albef, tcl（微调后）
├── third_party/
│   ├── albef/             # VE 微调
│   └── tcl/
├── tmm_scc/               # TMM + TMM-SCC 攻击核心
├── baselines/             # Co-Attack / SGA 适配层（baselines/co_attack/）
├── utils/                 # 实验逻辑
├── outputs/               # 白盒样本、黑盒结果、微调 checkpoint
├── prepare_subset.py      # 生成数据子集
├── run_attack.py          # 白盒攻击
├── run_ablation.py        # num_iters 消融
├── run_blackbox.py        # 黑盒迁移评测
├── finetune_ve.py         # VE 微调
└── run_all.py             # 流水线（可组合步骤）
```

## 环境

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 1. 生成 1000 图 VLR + 1000 条 VE 测试子集
python prepare_subset.py --name main_1k

# 2. 白盒攻击（ALBEF 替代模型，TMM + SCC）
python run_attack.py --task all --method all --subset main_1k --gpu 0

# 3. 黑盒评测（TCL + CLIP via HuggingFace，CLIP 无需本地权重）
python run_blackbox.py --task all --method all --subset main_1k

# 4. VE 微调（需先准备好 data/）
python finetune_ve.py --backbone albef --gpu 0
python finetune_ve.py --backbone tcl --gpu 0

# 5. num_iters 消融（小样本 ablation_200）
python prepare_subset.py --name ablation_200
python run_ablation.py --subset ablation_200 --gpu 0
```

## 一键流水线

```bash
# 默认: prepare → attack → blackbox
python run_all.py --subset main_1k --gpu 0

# 指定步骤
python run_all.py --steps prepare attack --subset main_1k

# 含消融
python run_all.py --steps prepare attack ablation blackbox --ablation-subset ablation_200

# 仅打印命令
python run_all.py --dry-run
```

## 输出约定

白盒结果：`outputs/whitebox/{task}/{model}/{method}/{subset}/`

- `adv_samples_ve/` 或 `adv_samples_retrieval/`
- `ve_adv_manifest.json` / `vlr_adv_manifest.json`
- `metrics.jsonl`

黑盒结果：`outputs/blackbox/results_log.jsonl`

微调结果：`outputs/finetune/{albef|tcl}/` → best 复制到 `checkpoints/VE/`

## 攻击方法

| 参数 | 说明 |
|------|------|
| `--text_method tmm` | 原版 TMM |
| `--text_method scc` | TMM-SCC（语义一致约束） |

全局超参见 `utils/config.py` 中 `AttackConfig`。

## CLIP 黑盒评测

CLIP 通过 HuggingFace 在线加载，无需放入 `checkpoints/`：

```python
from transformers import CLIPModel, CLIPProcessor
CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
```

黑盒评测时 `--target clip` 会自动使用上述模型。
