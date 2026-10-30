# 25-Lang-Model · 已训练声学模型

本仓库保存完成 **80,000次更新**的25语共享声学模型、音素时长/停顿/韵律预测、31/39维声学输出，以及独立训练和推理脚本。**不包含声码器网络或声码器权重**；波形生成的接入位置见 [声码器接口报告](docs/VOCODER_INTERFACE.md) 和 [`new25_acoustic/vocoder.py`](new25_acoustic/vocoder.py)。

声学模型为 **1,186,071个存储参数**。原约400万参数方案中的其余参数属于声码器，此次未发布。FP16张量净大小2,372,142字节；另提供FP32参考权重，便于核对和后续微调。

## 发布内容

| 内容 | 文件 |
|---|---|
| 已训练声学FP16权重，97个张量 | [`models/acoustic_fp16.safetensors`](models/acoustic_fp16.safetensors) |
| 同一声学状态的FP32参考权重 | [`models/acoustic_fp32.pt`](models/acoustic_fp32.pt) |
| 实际285音素共享词表 | [`models/vocabulary.json`](models/vocabulary.json) |
| 25语训练集归一化统计 | [`models/normalization.json`](models/normalization.json) |
| 模型结构、语言ID及维数 | [`models/model_config.json`](models/model_config.json) |
| 导出身份、逐文件SHA256 | [`models/EXPORT.json`](models/EXPORT.json) |
| 原80k检查点与发布张量逐项一致的证据 | [`receipts/acoustic_provenance.json`](receipts/acoustic_provenance.json) |
| 数据准备、声学训练及恢复 | [`docs/TRAINING.md`](docs/TRAINING.md) |
| 完成范围、训练来源和局限 | [`docs/ACOUSTIC_MODEL_CARD.md`](docs/ACOUSTIC_MODEL_CARD.md) |

语言ID顺序固定为：

```text
bg cs da de el en es et fi fr hr hu it lt lv nl no pl pt ro sk sl sv tr zh
```

ZH使用31维物理声学模式，其余24语使用39维；全部语言共用核心网络，保留小型语言嵌入和分模式投影。

## 安装与推理

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/verify_release.py
python -m unittest discover -s tests -v
```

已用Python 3.12.14 / PyTorch 2.6.0 / NumPy 1.26.4在CPU验证发布代码。原声学训练使用PyTorch 2.8.0、CUDA 12.8；详细配置见训练说明。CPU参考推理把FP16存储权重展开为FP32计算，不能据此声称完整Linux推理进程达到10 MB。

输入NPZ需要以下原前端特征，不包含音频：

| 数组 | 形状 / 类型 | 说明 |
|---|---|---|
| `phones` | `[N]`，整数 | 使用本仓库音素词表，不能套用其他词表的整数ID |
| `phonetic` | `[N,32]`或`[N,36]`，浮点 | 原前端特征，尚未归一化 |
| `context` | `[N,22]`或`[N,34]`，浮点 | 原上下文特征，尚未归一化 |
| `pause` | `[N]`，0/1 | 预测时长时必填；由文本/音素前端判定停顿候选 |
| `durations` | `[N]`，非负整数，可选 | 已有对齐时长，单位为10 ms帧 |

```bash
# 使用已有对齐时长；features.npz由兼容的原前端提供。
python -m new25_acoustic.inference \
  --features features.npz --language zh --model-dir models --output acoustic.npz

# 使用本模型预测时长/停顿；要求pause数组。
python -m new25_acoustic.inference \
  --features features.npz --language zh --model-dir models \
  --predict-durations --output acoustic_predicted_duration.npz
```

输出包含物理声学曲线、供声码器消费的归一化条件、实际使用的时长及显式声学流状态。采样率约定16 kHz，每帧160个采样点。模型支持整句编码后按帧分块解码；音素编码器需要整句上下文。

普通新文本到IPA及32/36维、22/34维特征的完整前端资产尚未交付。本仓库可消费已有兼容特征，不能把任意文本直接变成WAV。声码器接口也不会生成占位静音。

## 验收边界

80,000步完成回执证明声学训练达到本轮更新上限。发布核验覆盖原始检查点SHA、97个张量一致性、FP16舍入、独立模型前向、25语接口及分块状态。真实语音质量、原音色、噪声、完整任意文本TTS及整个Linux进程10,000,000字节峰值尚未全部验收。训练步数和权重文件大小不代替这些结果。

仓库包含模型及复现代码，不附带训练音频、训练特征、完整联合检查点、优化器历史、声码器或服务器运维文件。原模型和语料继续留在原位置，后续波形模型可按接口独立接入。
