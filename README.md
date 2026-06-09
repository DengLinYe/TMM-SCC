# TMM-SCC 实验框架

## 环境

```bash
pip install -r requirements.txt
```

## 统一入口

```bash
python main.py                 # 交互控制台
python main.py --quick         # 快速验证（等同菜单 0，--smoke 同义）
python main.py --full          # 正式流水线
```

## 交互菜单

| 选项 | 功能 |
|------|------|
| **0** | **快速验证全流程**（见下表，与 `--quick` 相同） |
| 1 | 生成数据子集 |
| 2 | 白盒攻击 |
| 3 | 黑盒评测 |
| 4 | num_iters 消融 |
| 5 | VE 微调 |
| 6 | 正式实验 (main_1k) |
| q | 退出 |

## 快速验证配置

与正式实验**步骤一致**，仅缩小规模：

| 项目 | 快速验证 | 正式实验 |
|------|----------|----------|
| 子集 | smoke_20（20 图 + 20 条 VE） | main_1k |
| num_iters | 2 | 5（config 默认） |
| 微调 epochs | 2（ALBEF + TCL） | 3 |
| 攻击方法 | TMM + SCC | TMM + SCC |
| 黑盒受害 | TCL + CLIP | TCL + CLIP |
| 消融 iters | 2, 3 | 3, 5, 10, 20 |

**执行顺序：** prepare → finetune(albef,tcl) → attack → ablation → blackbox

```bash
python main.py --quick --gpu 0
python main.py --quick --dry-run
```

## 目录

```
main.py          # 唯一入口
utils/           # 实验逻辑
tmm_scc/         # 攻击核心
third_party/     # ALBEF/TCL 微调
data/            # 数据
checkpoints/     # 权重
outputs/         # 输出
```

CLIP 黑盒从 HuggingFace 加载 `openai/clip-vit-base-patch16`，无需本地权重。
