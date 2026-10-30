# 当前分支：完整 25 语 INT8 推理交接包

分支 `int8/simulator-handoff-20261029` 在原 `3a7465f` 基础上追加当前全 INT8 权重、25 语文本前端、完整声学/梅尔适配器/HiFi-GAN 源码、数值对照工具，以及 M55 / RV32 模拟器入口；`main` 和原 FP16/FP32 文件保留。

**先读 [INT8 交接说明](docs/INT8_HANDOFF.md)**。普通文本到 WAV 已在 Linux 主机跑通全部 25 语。M55、RV32 已通过载入完整权重的真实层内核测试；整句原生推理调度与前端移植仍是合作者的优化工作，当前没有声称完整模型已在 MCU/QEMU 内从文本生成 WAV。

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-int8.txt
python scripts/verify_release.py
python scripts/verify_int8.py
python synthesize.py --language zh --text '你好，欢迎测试新的语音模型。' --output out/zh.wav
python synthesize.py --language en --text 'Hello, this is a test of the new voice model.' --output out/en.wav
# 只依赖 Python 标准库和已安装的 QEMU，无需 root、PyTorch 或编译器：
python scripts/run_simulator.py --target m55
python scripts/run_simulator.py --target rv32
```

- [`int8/weights.safetensors`](int8/weights.safetensors)：完整链 4,015,048 参数，4,154,768 字节；矩阵 INT8，逐输出通道 FP32 scale，偏置/归一化等保留 FP16。
- [`frontend/`](frontend/)：16 份 FEP 共 3,875,677 字节，规则、词典、Python/C 源码、Linux 库和归属记录齐全。运行时无需旧模型仓库、训练语料或联网下载字典。
- [`examples/host_smoke/`](examples/host_smoke/)：25 语普通文本的输入特征、完整 WAV 与校验记录；仅用于链路测试，不是质量通过证明。
- [`native/`](native/)、[`simulator/prebuilt/`](simulator/prebuilt/)：可替换的标量基准内核、双架构构建入口和已通过测试的预编译程序；权重可无损导出供原生移植。
- [`scripts/dump_trace.py`](scripts/dump_trace.py)：输出完整链逐模块数值，供模拟器实现逐层对齐。

以下保留原 FP16/FP32 发布说明，INT8 分支使用上面的新入口。

---

# 25-Lang-Model · 已训练声学模型与HiFi-GAN微调声码器

本仓库保存完成 **80,000次更新**的25语共享声学模型、音素时长/停顿/韵律预测、31/39维声学输出，以及独立训练和推理脚本。现已追加第二个实验完成 **31,403次Generator更新**的HiFi-GAN微调声码器及物理条件→mel适配器；可将已有兼容音素特征生成的声学结果合成为WAV。详见 [HiFi-GAN模型卡](docs/HIFIGAN_MODEL_CARD.md) 和 [整句推理说明](docs/HIFIGAN_INFERENCE.md)。

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

将完整声学结果转换为WAV：

```bash
python -m new25_hifigan.inference \
  --acoustic acoustic.npz --language zh \
  --model-dir models/hifigan --output output.wav
```

普通新文本到IPA及32/36维、22/34维特征的完整前端资产尚未交付。本仓库可消费已有兼容特征，经声学模型和HiFi-GAN生成音频；不能据此宣称任意新文本TTS已验收。HiFi-GAN当前为整句离线实现，未实现通用声码器接口的连续分块状态协议。

## 验收边界

80,000步完成回执证明声学训练达到本轮更新上限。发布核验覆盖原始检查点SHA、97个张量一致性、FP16舍入、独立模型前向、25语接口及分块状态。真实语音质量、原音色、噪声、完整任意文本TTS及整个Linux进程10,000,000字节峰值尚未全部验收。训练步数和权重文件大小不代替这些结果。

仓库包含声学模型、HiFi-GAN与适配器的model-only权重和复现/推理代码；不附带训练音频、训练特征、完整联合训练检查点、优化器历史、判别器权重或服务器运维文件。HiFi-GAN的原MIT许可保留于 [`third_party/hifi_gan/LICENSE`](third_party/hifi_gan/LICENSE)。
