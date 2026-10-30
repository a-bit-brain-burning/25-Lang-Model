# 声码器接口规格与实现交接

本仓库交付 25 语言声学模型及声码器接入位置。声码器的网络、训练代码和权重均未交付；`VocoderStub` 会抛出 `NotImplementedError`，不会生成占位音频。实现入口是 [`new25_acoustic/vocoder.py`](../new25_acoustic/vocoder.py)，声学结果由 [`new25_acoustic/inference.py`](../new25_acoustic/inference.py) 产生。

接口目标是把本版本的 **31/39 维物理声学表示**转换成同一时间线上的 16 kHz 单声道浮点 PCM。这些谱形分量的原始基底和标度尚未完整说明，不能将其命名为通用 mel、线性频谱或某个已知倒谱格式，也不能直接接到仅接受 mel 的预训练声码器。

## 1. 调用约定

实际 `Protocol` 使用 Python `__call__`；下面的调用即“消费条件帧”的操作：

```python
pcm, next_state = vocoder(normalized, physical, lang, state=None)
```

| 参数/返回值 | 形状、类型 | 约定 |
|---|---|---|
| `normalized` | 浮点 Tensor `[B,T,D]` | 从预测的物理条件按本版本、对应语言的均值/标准差计算，见第 3 节 |
| `physical` | 浮点 Tensor `[B,T,D]` | 第 2 节的物理条件；与 `normalized` 对应同一帧、同一语言 |
| `lang` | `torch.long` Tensor `[B]` | 固定语言 ID，见第 4 节；不是字符串，也不是任意重排后的索引 |
| `state` | `Any` 或 `None` | 外部声码器的私有流状态；`None` 表示新语句 |
| `pcm` | 浮点 Tensor `[B,T*160]` | 16 kHz、mono；时间维不是 `[B,1,S]` 或 `[B,S,1]` |
| `next_state` | `Any` | 下一块继续使用的声码器状态 |

`B>=1`、`T>=1`，`D` 只能为 31 或 39。三个输入 Tensor 使用同一 device，数值必须有限；`physical[..., -1]` 必须在 `[0,1]`。同一批次只有一个 `D`，因此不能把 31 维的 `zh` 与 39 维语言直接放在一个稠密批次中。发布的 `AcousticPredictor` 目前处理单语句，正常交接是 `B=1`。

使用 `validate_condition(normalized, physical, lang)` 校验形状、类型、device、有限值、语言与 `D` 的对应关系及物理 voicing 范围；使用 `validate_output(pcm, normalized)` 校验输出形状和有限值。这两个检查不证明归一化数值、音色或音质正确，适配器仍须验证其统计文件和条件语义。

参考推理返回 FP32 数组。外部实现可以选择其他浮点计算精度，但须自行验证误差与状态精度；本接口没有对所有后端作 FP16 数值或性能承诺。浮点 PCM 以幅度 1 表示满幅，接口未强制硬限幅，调用方应测量超幅率。写入 PCM16 文件时再按明确策略限幅、量化，不能直接整数转换导致溢出。

## 2. 帧内通道定义

下表使用 **从 0 开始、两端包含**的索引；代码切片右端不包含。

| 语义 | `D=31`，用于 `zh` | `D=39`，用于其他 24 语言 | 数值说明 |
|---|---|---|---|
| `log_f0` | `0` | `0` | F0 Hz 的自然对数；不是 log10、音分或 pitch period |
| `log_energy` | `1` | `1` | 原 waveform mean-square 的自然对数；不是 RMS、dB 或线性能量 |
| 原谱形分量 | `2..24`，共 23 个；`[...,2:25]` | `2..32`，共 31 个；`[...,2:33]` | 保持原表示的顺序、基底和标度；当前不能假定为 mel |
| `AP5` | `25..29`；`[...,25:30]` | `33..37`；`[...,33:38]` | 五个非周期性分量，真实标签约在 `[-1,0]`；频带边界及具体变换尚未完整确认 |
| 物理 voicing | `30` | `38` | `[0,1]` 的连续强度/概率；不是 logit，也不是强制二值掩码 |

源契约写作 `S=D-7`，这里 `S` 是谱形部分的**最后一个索引**：`2..S`，共 `D-8` 个谱形分量。不要据此误读成 24/32 个谱形分量。

真实参考标签中，无效/无声的 `log_f0` 可以为 0；它表示“无有效 F0”，不能据 `exp(0)` 解释成有效 1 Hz。真实标签的 F0 有效性来自独立的 `frame_f0_valid`，不是由 `log_f0==0` 或某个 voicing 阈值推断。已确认的有效 F0 标签范围是 40–600 Hz。预测路径仍保留连续预测的 `log_f0`，不能先根据 `voicing<0.5` 将其清零或替换均值。

源波形实现为计算 pitch lag，会将物理 `log_f0` 钳制到 `ln(40)..ln(600)`，将 `log_energy` 钳制到 `[-28,0]`，并用连续 voicing 控制反馈强度。这说明原条件的解释方式；本仓库没有提供该波形网络。外部适配器如使用不同钳制、带宽映射或 voicing 策略，应记录并验证。声学结果本身未对所有物理通道作上述限幅，预测 AP5 也可能超出真实标签的典型范围。

## 3. 声学头 → 物理条件 → 声码器归一化条件

从 [`models/normalization.json`](../models/normalization.json) 取当前语言的 `mean[D]` 和 `std[D]`。它们来自训练划分，不能用测试语音重新拟合。记声学头输出为 `z[B,T,D]`：

```python
physical = z.float() * std + mean
physical = torch.cat([
    physical[..., :-1],
    torch.sigmoid(z[..., -1:]),
], dim=-1)
normalized = (physical - mean) / std
```

关键区别是：`z[..., :-1]` 预测归一化的物理通道；`z[..., -1]` 预测 **voicing logit**。末维直接对该 logit 做 sigmoid，不能先做均值/标准差反变换后再 sigmoid。生成 `physical` 后，必须对**全部通道**按均值/标准差得到 `normalized`，包括末维，不能把原始 `z` 整体当成声码器的 `normalized`。

本次归一化文件中，每种语言的末维恰为 `mean[-1]=0`、`std[-1]=1`，所以声码器归一化条件的末维数值恰好也是 sigmoid 概率。实现仍应保留上述通式，避免误把这一特例当成“直接传 logit”的理由。`AcousticResult.physical` 和 `.normalized` 已完成这两步，接入时无需再次 sigmoid 或重复归一化。

真实标签回放有一个额外区别：源数据加载器会把 `frame_f0_valid=False` 的 F0 **仅在归一化副本中**替换为训练均值，使该维归一化后为 0；原物理标签仍可能保存 0。此时真实标签回放的两组条件在无效 F0 维不必满足逐元素的 `(physical-mean)/std`。该补值策略只用于读取带有效性标签的真实条件，不能搬到连续预测路径。当前公开的 `AcousticPredictor` 只输出预测条件。

## 4. 固定语言顺序

语言顺序来自训练时的语言 embedding，不能按显示名称或本地字母顺序重新生成。

```text
ID:   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
lang: bg cs da de el en es et fi fr hr hu it lt lv nl no pl pt ro sk sl sv tr zh
```

`lang==24` 对应 `D=31`，其余对应 `D=39`。应优先从 [`models/model_config.json`](../models/model_config.json) 或 `new25_acoustic.inference.LANGUAGES` 读取，且与权重、词表和归一化文件作为同一版本部署。

## 5. 时间线与状态

采样率为 `16000 Hz`，hop 为 `160 samples`，即每帧 `10 ms`、每秒 `100` 帧。第 `t` 个条件帧对应输出样本区间 `[160*t,160*(t+1))`，`T` 帧必须输出恰好 `160*T` 个样本。不添加历史版本的 80 样本偏移、隐式头尾裁剪、预加重或去加重，也不自动改变采样率。

声码器须遵守以下状态生命周期：

1. 每个独立语句以 `state=None` 开始；同一语句连续块按原顺序传入，保留每次返回的 `next_state`。
2. 新语句、语言切换、31/39 维模式切换、批次元素顺序改变或批次大小改变时重置状态。若要独立重置某个批次元素，由外部实现提供明确方法。
3. 不在普通分块边界、音素边界或静音处自行重置。声码器私有状态与 `AcousticResult.acoustic_state` 是两个不同对象，不能互传。
4. 接口无 `lengths`/padding mask，传入的帧全部被当成有效帧。长度不同的语句须分别调用，或在外层明确处理补齐和输出裁切。
5. 每次调用立即返回该块的 `T*160` 样本。需要未来帧、额外暖启动或尾部 flush 的模型，须由适配层管理缓冲并明确延迟；未经适配不能宣称满足此逐块接口。

声学解码器支持显式帧状态，`chunk_frames` 在完整句子的编码与帧布局完成后切分解码。然而音素编码器包含非因果卷积，依赖左右音素上下文。不能将输入 token 分成独立小句并分别 `encode`，再把结果拼接当作完整句子的等价推理。分块时 `frame_phone` 仍引用完整音素序列，`position` 仍按该音素的完整时长计算。

## 6. 对接示例

以下示例放在仓库根目录运行。`frontend_input.npz` 由已验证的原前端产生；`my_vocoder` 必须是实现本协议的真实外部对象。本仓库没有随附该对象。

```python
import numpy as np
import torch
from new25_acoustic import AcousticPredictor
from new25_acoustic.inference import LANGUAGES
from new25_acoustic.vocoder import validate_condition, validate_output

predictor = AcousticPredictor.from_pretrained("models", device="cpu")
with np.load("frontend_input.npz", allow_pickle=False) as f:
    result = predictor.predict(
        phones=f["phones"], phonetic=f["phonetic"], context=f["context"],
        pause=f["pause"], language="zh", chunk_frames=32,
    )

# AcousticResult 中的数组形状是 [T,D]，必须显式添加 batch 维。
normalized = torch.from_numpy(result.normalized).unsqueeze(0)
physical = torch.from_numpy(result.physical).unsqueeze(0)
lang = torch.tensor([LANGUAGES.index(result.language)], dtype=torch.long)
state, blocks = None, []
with torch.inference_mode():
    for start in range(0, physical.shape[1], 32):
        n, p = normalized[:, start:start+32], physical[:, start:start+32]
        validate_condition(n, p, lang)
        pcm, state = my_vocoder(n, p, lang, state)
        validate_output(pcm, n)
        blocks.append(pcm)
pcm = torch.cat(blocks, dim=1)
assert pcm.shape[1] == result.physical.shape[0] * 160
```

`result.prediction` 保存原声学头输出，不能代替 `.normalized`。如选择 CUDA 或其他外部后端，需要显式把两个条件 Tensor、`lang` 及声码器移到一致的 device。该示例只展示交接，未验收音质。

## 7. 外部实现建议与验收项目

实现可训练直接消费该 31/39 维表示的声码器，或训练一个有成对监督的映射器，再接到外部模型。31 维和 39 维需各自定义真实映射；不能靠截断、补零、重复通道或仅更改张量宽度声称兼容。若转为某种 mel 表示，应先补齐原谱形基底、AP5 定义和分析窗口等元数据，或者使用合法获取的配对物理条件/PCM 学习并验证映射。相同维数和采样率不足以证明兼容。

在宣布实现满足接口前，至少应保存以下可复核结果：

- **数值与语义**：固定语言 ID；对应语言、维数及统计文件；logit/sigmoid/再归一化数值检查；F0 无效标签与连续预测路径分别检查。
- **时间与流式**：多种非整除块长、单帧块和跨音素边界，验证总样本数；整句与分块在声明容差内一致；新语句重置后不受上一句影响；首尾不隐式位移。
- **稳健性**：静音、低 voicing、F0 边界、长句、异常有限值/形状、状态重用错误；记录 NaN/Inf、超幅率、尾部漂移和接口延迟。
- **实际能力**：用明确划分的语料检查 25 种语言，并另行开展听感评价。频谱误差、短样例可播放或接口检查通过都不构成 MOS 分数。
- **资源**：在目标 Linux 环境测量完整单进程峰值 RAM、前端、声学模型、声码器、状态、运行时与缓冲；FP16 权重字节数不能替代整个进程的 10,000,000 字节预算验收。

## 8. 已核对的依据与易错点

接口依据为来源归档中 `local_previews/inflight_cpu_preview_v3_20261030/payload/` 下的相对路径。归档本身未作为声码器或训练工程随此仓库交付。

| 来源相对路径 | SHA-256 | 核对内容 |
|---|---|---|
| `source/newtts/model.py` | `22c9a050b5204cfb382c38a656382f31423e822ac8d6f0039da0a74ca4fe01c3` | 31/39 头、非因果编码器、因果帧解码器、原波形消费签名及时间线 |
| `source/newtts/data.py` | `17de19edf87c50e34309cd9dee181897c7058f04284746d08607661ccdf5b300` | 末维直接 sigmoid、全维再归一化、真实无效 F0 仅加载时补值、帧位置 |
| `source/newtts/contracts.py` | `0d51b63ecb4a5836e1b88001e21fef08959b9e7535bb0d1477c2400d0d4cda8e` | 语言顺序、16k/160、真实 AP5/voicing/F0 标签验证 |
| `source/tools/evaluate.py` | `96e864dc40bd722854b40c06b306f925ecd5262badf62a3a3e5e82ede60304fa` | 预测条件交接、32 帧流式保留状态、预测时长和参考前端能力边界 |
| `source/docs/DATA_CONTRACT.md` | `576831f45208a7f3f701428df7c852a3ed23897cb521ab1cc3345b73003b730a` | 谱形/AP5 索引、F0/能量语义及原文本前端尚未交付 |
| 本仓库 `models/normalization.json` | `b5db6854811557e26da1471ab22b1d8daed3dc63a22eb108d08cb13a9c4909b2` | 按语言统计维数；末维 mean=0、std=1 的本次特例 |

已特别核对谱形的“索引上界”与“数量”区别、物理概率与 logit 区别、头输出与声码器归一化条件区别、无效真实 F0 与预测 F0 区别、完整句编码与帧流式区别，以及声学状态与声码器状态不能混用。声学权重和验收边界见 [`ACOUSTIC_MODEL_CARD.md`](ACOUSTIC_MODEL_CARD.md)。
