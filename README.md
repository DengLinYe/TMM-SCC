# TMM-SCC 实验框架

## 环境

依赖见 `requirements.txt`，安装完成后即可运行：

```bash
pip install -r requirements.txt
```

## 运行

```bash
python main.py
```

GPU 编号在 `utils/config.py` 的 `GPU` 项中配置。

## 控制台选项

| 选项    | 说明                                                |
| ------- | --------------------------------------------------- |
| **1–5** | 分步执行：子集 / 白盒 / 黑盒 / 消融 / 微调          |
| **6**   | 本地测试全流程（`mini_100`，`iters=2`）             |
| **7**   | 本地正式全流程（`main_1k`，`iters=10`）             |
| **8**   | 强制测试（同 6，强制重跑 VE 微调）                  |
| **9**   | 云端测试全流程（`mini_100`）                        |
| **10**  | 云端消融（`ablation_200`）                          |
| **11**  | 云端正式全流程（`main_1k`）                         |
| **12**  | 云端正式全流程（无微调，复用已有 `checkpoints/ve`） |
| **q**   | 退出                                                |

### 云端推荐顺序

```
9 → 10 → （根据消融结果修改 config 中 attack_num_iters）→ 11
```

## 目录结构

```
main.py          # 入口
utils/config.py  # 参数与 RunProfile
tmm_scc/         # 攻击与黑盒评测
checkpoints/     # 权重与 clean_metrics.json
outputs/         # 攻击产物与 run.json
data/            # 数据集标注
```
