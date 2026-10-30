# 25-Lang-Model · 已训练声学模型、HiFi-GAN、FARGAN与RNDVoC声码器

本仓库保存完成 **80,000次更新**的25语共享声学模型、音素时长/停顿/韵律预测、31/39维声学输出，以及独立训练和推理脚本。现已追加第二个实验完成 **31,403次Generator更新**的HiFi-GAN微调声码器及物理条件→mel适配器；可将已有兼容音素特征生成的声学结果合成为WAV。详见 [HiFi-GAN模型卡](docs/HIFIGAN_MODEL_CARD.md) 和 [整句推理说明](docs/HIFIGAN_INFERENCE.md)。

另新增固定开发窗口选中的 **4750步FARGAN及31/39维→20维adapter**，复用本仓库已有80k声学模型。完整FARGAN链为 **3,895,008参数、7,790,016字节FP16净权重**；新增文件仅含FARGAN和adapter。其离线入口保留原有效中心`[400,160*T−240)`，输入T帧输出`(T−4)*160`采样，并写对齐sidecar；详见[FARGAN模型卡](docs/FARGAN_MODEL_CARD.md)和[推理说明](docs/FARGAN_INFERENCE.md)。

现已追加 **25,000步RNDVoC-Lite本地复现候选**及同一E2 mel adapter。RNDVoC为713,929参数，接原80k声学模型和adapter后共 **3,802,992个共享可学习参数**。默认可学习权重FP16、三个非学习投影缓冲FP32；它是查询时已评估的开发集最佳，不表示最终训练结束。真实mel重建SC为0.173447，80k预测特征＋真实时长整链SC为0.717298，两者不可混用。详见[RNDVoC模型卡](docs/RNDVOC_MODEL_CARD.md)和[接入接口](docs/RNDVOC_INFERENCE.md)。

另提供独立的 **Model原Opus BWE24后端**：16k波形经原16→48k扩带和对齐流程输出24k，新增367,890参数；72个原始资产字节与许可保持。支持在声码器与扩带之间通过`stages=[...]`插入其他真实16k波形模块，详见[模块插入协议与示例](docs/POSTPROCESS_MODULES.md)。仓库声码器权重仍为25k；已有35k＋BWE试听作为先前运行证据单列，本次没有替换权重或重新生成试听。

声学模型为 **1,186,071个存储参数**，FP16张量净大小2,372,142字节。追加的HiFi-GAN Generator为925,985参数、适配器为1,902,992参数，**完整链合计4,015,048参数、8,030,096字节FP16净权重**。两部分均提供FP16部署权重和FP32参考权重。

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
| HiFi-GAN与适配器的FP16权重 | [`models/hifigan/vocoder_adapter_fp16.safetensors`](models/hifigan/vocoder_adapter_fp16.safetensors) |
| HiFi-GAN与适配器的FP32主权重 | [`models/hifigan/vocoder_adapter_fp32.pt`](models/hifigan/vocoder_adapter_fp32.pt) |
| 声码器来源、配置及文件身份 | [`models/hifigan/EXPORT.json`](models/hifigan/EXPORT.json) |
| 声码器推理和能力边界 | [`docs/HIFIGAN_INFERENCE.md`](docs/HIFIGAN_INFERENCE.md)、[`docs/HIFIGAN_MODEL_CARD.md`](docs/HIFIGAN_MODEL_CARD.md) |
| FARGAN 4750与adapter的FP16权重，91张量 | [`models/fargan/vocoder_adapter_fp16.safetensors`](models/fargan/vocoder_adapter_fp16.safetensors) |
| FARGAN身份、声学依赖及SHA256 | [`models/fargan/EXPORT.json`](models/fargan/EXPORT.json) |
| FARGAN有效中心推理和能力边界 | [`docs/FARGAN_INFERENCE.md`](docs/FARGAN_INFERENCE.md)、[`docs/FARGAN_MODEL_CARD.md`](docs/FARGAN_MODEL_CARD.md) |
| RNDVoC25k权重，FP16参数＋FP32缓冲 | [`models/rndvoc/vocoder_fp16.safetensors`](models/rndvoc/vocoder_fp16.safetensors) |
| RNDVoC25k未舍入FP32参考权重 | [`models/rndvoc/vocoder_fp32.safetensors`](models/rndvoc/vocoder_fp32.safetensors) |
| 与已有E2完全相同的独立adapter权重 | [`models/rndvoc/adapter_fp16.safetensors`](models/rndvoc/adapter_fp16.safetensors) |
| RNDVoC身份、逐张量来源及文件SHA | [`models/rndvoc/EXPORT.json`](models/rndvoc/EXPORT.json)、[`models/rndvoc/TENSOR_PROVENANCE.json`](models/rndvoc/TENSOR_PROVENANCE.json) |
| RNDVoC原75句评估和提交静态核对 | [`receipts/rndvoc_development_summary.json`](receipts/rndvoc_development_summary.json)、[`receipts/rndvoc_release_static_validation.json`](receipts/rndvoc_release_static_validation.json) |
| 原Model Opus BWE后端和中间模块插槽 | [`new25_postprocess/bwe24.py`](new25_postprocess/bwe24.py)、[`new25_postprocess/chain.py`](new25_postprocess/chain.py) |
| 原件源码、静态C权重和SHA | [`new25_postprocess/assets/opus_bwe/`](new25_postprocess/assets/opus_bwe/)、[`BWE_ASSET_MANIFEST.json`](new25_postprocess/assets/BWE_ASSET_MANIFEST.json) |
| 插入位置、模块契约与使用方法 | [`docs/POSTPROCESS_MODULES.md`](docs/POSTPROCESS_MODULES.md) |
| 既有运行证据和本次静态检查 | [`receipts/opus_bwe24_provenance.json`](receipts/opus_bwe24_provenance.json)、[`receipts/postprocess_release_static_validation.json`](receipts/postprocess_release_static_validation.json) |

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
# 使用新增HiFi-GAN离线声码器时，另安装SciPy依赖：
python -m pip install -e '.[hifigan]'
# FARGAN使用同一SciPy依赖，也可选择：
python -m pip install -e '.[fargan]'
# RNDVoC另需librosa构造固定mel投影：
python -m pip install -e '.[rndvoc]'
python scripts/verify_release.py
python scripts/verify_fargan.py
# 此入口仅解析源码、元数据和权重文件，不导入模型或执行推理：
python scripts/verify_rndvoc.py
python scripts/verify_postprocess.py
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

将完整声学结果转换为WAV：

```bash
python -m new25_hifigan.inference \
  --acoustic acoustic.npz --language zh \
  --model-dir models/hifigan --output output.wav
```

选择FARGAN 4750的原有效中心路径（输出同时生成`output_fargan.wav.json`对齐说明）：

```bash
python -m new25_fargan.inference \
  --acoustic acoustic.npz --language zh \
  --model-dir models/fargan --seed 0 --output output_fargan.wav
```

FARGAN要求至少5帧，输出从原时轴第400个采样开始，结束于`160*T−240`。不补首尾，不能用完整原句长度或逐块`T*160`规则解释；它与HiFi-GAN是两种独立离线声码器选择。4750的75句开发评分使用真实时长，不能当作预测时长路径的评分。

选择RNDVoC25k完整时间线（T帧输出恰好T*160个16k采样）：

```bash
python -m new25_rndvoc.inference \
  --acoustic acoustic.npz --language zh \
  --model-dir models/rndvoc --output output_rndvoc.wav
```

RNDVoC还提供`RNDVoCVocoder.synthesize_mel(raw_logmel)`，直接消费固定HiFi-GAN帧网格的`[1,80,M]`自然对数mel，返回22.05k的`[1,M*256]`PCM。当前是CPU整句参考实现；不提供连续声码器状态。全链FP16可学习张量7,605,984字节，另有1,380,996字节FP32投影缓冲，均不能替代完整10MB进程实测。

追加原件24k输出（首次实际使用需C编译器，原件离线构建；普通导入及静态检查不构建）：

```bash
python -m pip install -e '.[rndvoc,postprocess]'
python -m new25_rndvoc.inference \
  --acoustic acoustic.npz --language zh --model-dir models/rndvoc \
  --bwe24 --output output_24k.wav
```

可选链路为`RNDVoC 16k → 模块A → 模块B → 原Opus BWE24 → 24k`。中间模块均须保持16k单声道PCM16和原长度，每句重置状态；输入为mel或物理声学特征的模块应接在声码器前。新增插槽只做静态接口核对，外部模块的真实组合效果另需验收。

普通新文本到IPA及32/36维、22/34维特征的完整前端资产尚未交付。本仓库可消费已有兼容特征，经声学模型和HiFi-GAN生成音频；不能据此宣称任意新文本TTS已验收。HiFi-GAN当前为整句离线实现，未实现通用声码器接口的连续分块状态协议。

## 验收边界

80,000步完成回执证明声学训练达到本轮更新上限。发布核验覆盖原始检查点SHA、97个张量一致性、FP16舍入、独立模型前向、25语接口及分块状态。真实语音质量、原音色、噪声、完整任意文本TTS及整个Linux进程10,000,000字节峰值尚未全部验收。训练步数和权重文件大小不代替这些结果。

仓库包含声学模型、HiFi-GAN、FARGAN、RNDVoC与相应适配器的model-only权重和复现/推理代码；不附带训练音频、训练特征、完整联合训练检查点、优化器历史、判别器权重或服务器运维文件。HiFi-GAN的原MIT许可保留于 [`third_party/hifi_gan/LICENSE`](third_party/hifi_gan/LICENSE)，Opus FARGAN原许可及IPR说明保留于[`third_party/opus/`](third_party/opus/)，RNDVoC上游MIT原文及固定来源保留于[`third_party/rndvoc/`](third_party/rndvoc/)。

RNDVoC本次提交复用已有真实75句评估/试听回执，只执行模型张量序列化、源码与接口静态核对；没有为push重跑训练、推理、数据集评测或性能实验。静态检查通过不表述为新发布入口的运行验证通过。

BWE原件及插入接口的提交同样仅做静态检查，复用此前Linux x86_64构建、原件数值/长度/状态检查及35k试听回执；本次未运行C构建、推理或实验。原件BSD许可及来源保留在[`assets/opus_bwe/licenses/`](new25_postprocess/assets/opus_bwe/licenses/)。

## FARGAN 流式推理与 M55 原生加速（2026-10-30）

本分支在 `main` 的离线 FARGAN 4750 接口之上追加两条已验证内容，训练更新为 **0**，原 91 个 FARGAN 张量及声学/HiFi-GAN/RNDVoC/Opus BWE 权重不变：

1. **主机连续流式**：`new25_fargan.FARGANVocoder.streaming=True`，`begin_stream`/`synthesize_chunk` 跨块保留 adapter 历史、3×GRU、FWConv、激励历史、独立 RNG 和去预加重状态；`synthesize_offline()` 保留原整段参考。输出仍为原有效中心 `(T-4)*160`（原时轴 `[400,160*T-240)`），`model_config.json` 历史 `streaming:false` 按文件身份保留。`freeze_weight_norm` 仅折叠 6 个重复 legacy hook，PCM 与未优化逐字节相同。新增流式契约单元测试后全套 **29 项通过**。入口与回执见 [`pipelines/fargan_streaming_20261030/`](pipelines/fargan_streaming_20261030/)，试听见 [`examples/fargan_streaming/listening/`](examples/fargan_streaming/listening/)。
2. **M55 原生加速**：已认可量化音质基线 `i8_final`（中文 ELF `87e096d0…`，同核顺序 QEMU 首有效 PCM24 3.6507→2.2601 s，-38.09%），以及当前最快且六输出逐字节一致的指令候选 `opt300d opt10`（62,678,844 动态指令事件，-8.919%，1 IPC 下界 313.394 ms）。二者源码、构建输入、QEMU 报告、动态 profile 与试听见 [`native/fargan_streaming_accel_20261030/`](native/fargan_streaming_accel_20261030/)。

**边界**：QEMU 主机秒数不是 200 MHz 周期；完整原生 25 语前端、完整系统双 10 MB、校准后的 200 MHz 首音 300 ms、任意长文与全面人工音质验收均未通过；`opt300d` 为未最终化实验候选，未替换默认生产指针。详见 [`docs/FARGAN_STREAMING_ACCELERATION.md`](docs/FARGAN_STREAMING_ACCELERATION.md)。
