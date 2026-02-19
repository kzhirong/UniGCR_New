统一生成式检索 (Generative Retrieval, GR) 与点击率预测 (CTR) 的多任务学习框架。其核心思想是利用共享的序列表达能力，同时驱动两种不同性质的推荐任务。


```Text
UniGCR_Repo/
├── ds_config.json          # DeepSpeed 配置文件
├── requirements.txt        # 依赖列表 (含安装顺序说明)
├── run.py                  # 启动入口
└── src/
    ├── __init__.py
    ├── config.py           # 全局配置 (Dataclass)
    ├── data.py             # UniversalDataset & DataLoader
    ├── grid_utils.py       # Semantic ID 映射与反查工具
    ├── model.py            # 模型核心 (InputLayer, HSTU, Heads)
    ├── trainer.py          # 训练循环, EarlyStop, Eval
    └── utils.py            # Metrics, Distributed Utils


配置
PyTorch 带CUDA
# 示例：安装 PyTorch 2.1 + CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
安装Flash Attention 2
pip install packaging ninja
pip install flash-attn --no-build-isolation

安装其他依赖包
# 安装 DeepSpeed, Scikit-learn 等
pip install deepspeed numpy pandas scikit-learn tqdm wget triton

# 安装 Meta 的 generative-recommenders (HSTU)
pip install git+https://github.com/facebookresearch/generative-recommenders.git@main


必要的准备工作 (Checklist)
在运行之前，请确认以下文件存在：
GRID Mapping File: data/beauty/semantic_ids.json。这是由 GRID 预处理生成的，格式应为 { "item_id": [code1, code2, code3], ... }。
Config Adjustments: 在 run.py 中，请务必修改 conf.num_atomic_items 为你数据集真实的 Item 总数，否则 Embedding 层会报错或越界。
# 指定可见设备
# 即使是单卡，也建议用 torchrun 启动以保持环境一致
torchrun --nproc_per_node=1 run.py \
    --deepspeed \
    --deepspeed_config ds_config.json \
    --grid_mapping data/beauty/semantic_ids.json

评价指标完善：
GR: 保持了 HitRate 和 NDCG。
CTR: 新增了 AUC 和 LogLoss。通过 gather_tensors 确保了在多 GPU 环境下，AUC 是基于全局数据计算的，而不是局部 AUC 的平均值（那是不准确的）。


