# 基于大模型语义一致约束的 TMM

> - 从 `github.com/whdii/TMM.git` 修改而来，原版本至少在我本地无法正常运行，主要做了三部分的前期工作：
>   1. 修复导致无法运行的诸多bug，其中大部分是使用未定义的成员变量、成员函数，即原作者可能没有实现部分函数；
>   2. 同步依赖为更新版本的环境：Python3.12、CUDA13.0、PyTorch2.11、Transformers 4.38.2，并解决大量兼容性问题；
>   3. 增加了VE下游任务相关的代码，包括`models\model_ve.py`与`EvalVEAttack.py`等。
> - 构建了TMM-SCC核心方法，主要体现在`attack\textAttackSCC.py`文件中，也涉及部分其他文件改动。
> - 为论文结果产出搭建了一系列脚本，包括数据集的随机抽样、攻击实施、黑盒攻击迁移以及超参数分析。

## 一、环境依赖

### 1.1 运行环境

1. 基于Windows11 + RTX50系列显卡平台，配套环境版本较新。

2. 完全采用Python，其环境依赖如下：

   ```
   # PyTorch with CUDA 13.0
   --extra-index-url https://download.pytorch.org/whl/cu130
   torch
   torchvision
   
   # Base Dependencies
   ruamel.yaml
   pillow
   requests
   tqdm
   scipy
   setuptools<82
   protobuf
   opencv-python
   
   # NLP & Model Weights
   tokenizers>=0.19
   transformers>=4.38.2
   timm==0.4.9
   bert_score==0.3.13
   sentence-transformers
   spacy
   openai
   
   # Computer Vision Metrics
   scikit-image
   pytorch-msssim
   ```

### 1.2 实验数据与模型

1. 实验数据均来自公开数据集：

   ```
   # Flickr30K-image：
   https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
   
   # Flickr30K-label：
   https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json
   
   # SNLI-VE-image：
   同Flickr30K
   
   # SNLI-VE-label：
   https://github.com/salesforce/ALBEF
   ```

2. 实验模型：

   1. VLR任务模型均直接下载自官方Github仓库下载链接

      ```
      # ALBEF
      https://github.com/salesforce/ALBEF
      
      # TCL
      https://github.com/uta-smile/TCL
      ```

   2. VE任务对应模型官方均未公布微调权重，是我自己进行微调的。

   

## 二、脚本说明

```
# 直接run即可
test_cuda		仅测试CUDA环境
run_vlr_attack	自动化生成TMM与TMM-SCC的针对VLR任务的对抗样本
run_ve_attack	自动化生成TMM与TMM-SCC的针对VE任务的对抗样本
run_ablation	自动执行对num_iters参数影响分析
test_scc_api	仅测试本文核心方法，即大模型生成对抗文本的接口连通性
cut_dataset		随机抽样数据集的测试子集，原理是对标注文件的切割
EvalBlackBox	自动化迁移黑盒攻击脚本

# 复杂命令
EvalTransferAttack 进行白盒攻击并生成VLR任务对抗样本：

python EvalVEAttack.py --adv 1 --gpu 0 `
--checkpoint ./checkpoints/albef_ve_snli_ve.pth `
--config ./configs/ve_snli-ve.yaml `
--save_json_name result.json `
--config_name run_config `
--text_method scc --sim_threshold 0.65 `
--output_dir ./output/VE_SCC/final `
--save_dir ./output/VE_SCC/final/ `
--log_name result 

EvalVEAttack 进行白盒攻击并生成VE任务对抗样本

python EvalVEAttack.py --adv 1 --gpu 0 `
--checkpoint ./checkpoints/albef_ve_snli_ve.pth `
--config ./configs/ve_snli-ve.yaml `
--save_json_name result.json `
--config_name run_config `
--text_method tmm `
--output_dir ./output/VE_TMM/final `
--save_dir ./output/VE_TMM/final/ `
--log_name result 
```

