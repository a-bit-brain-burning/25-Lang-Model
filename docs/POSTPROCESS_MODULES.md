# 原Model扩带后端与模块插入方法

本仓库将已验证的Model原件作为独立`new25_postprocess`包发布，保留原Opus1.6 / BBWENet v2的72个源码、生成权重、许可和SHA。新增模块只包装入口与调用顺序，不改变原BWE特征、网络、34个48k采样延迟补偿、尾补640输入采样或24k重采样规则。

## 插入位置

```mermaid
flowchart LR
  A[80k声学模型] --> B[E2 adapter]
  B --> C[RNDVoC]
  C -->|16k PCM16| D[可选模块A]
  D -->|16k PCM16| E[可选模块B]
  E -->|16k PCM16| F[原Opus BWE24]
  F --> G[24k PCM16 WAV]
```

这里是**声码器之后、扩带之前的波形处理位置**。降噪、波形修复等模块可以接入；每个模块必须输出同时间线、同长度的16k单声道PCM。处理31/39维物理特征或80维mel的声学模块应放在声码器前；如一定要放在波形位置，应另写并验收PCM分析／重建适配器，不能把PCM直接当成声学特征输入。

## 直接使用原件

```bash
python -m pip install -e '.[rndvoc,postprocess]'

# 使用仓库已有25k声码器；默认仍输出16k，加此选项才启用扩带。
python -m new25_rndvoc.inference \
  --acoustic acoustic.npz --language zh --model-dir models/rndvoc \
  --bwe24 --output output_24k.wav

# 已有16k单声道PCM16 WAV，也可单独调用后端。
python -m new25_postprocess.bwe24 \
  --input existing_16k.wav --output output_24k.wav

# 纯静态检查；不导入组件、不构建、不生成音频。
python scripts/verify_postprocess.py
```

首次实际构造`OpusBWE24()`需要C编译器；有CMake时按原CMake配置以2个编译worker离线构建，无CMake时支持等价便携标量源码清单。无需下载模型。缓存默认在系统临时目录，也可用`--postprocess-cache`指定仓库和原件资产之外的目录。构建不发生在普通包导入或静态检查时。

SDK直接调用：

```python
from new25_rndvoc import RNDVoCVocoder
from new25_postprocess import OpusBWE24, RNDVoCWithBWE24

vocoder = RNDVoCVocoder.from_pretrained("models/rndvoc")
backend = RNDVoCWithBWE24(vocoder, OpusBWE24())
pcm24 = backend.synthesize(normalized, "zh")
# normalized是原AcousticResult的[1,T,31/39]条件；不是PCM。
# pcm24是numpy.float32[1,N24]，采样率24000Hz。
report = backend.last_report
```

声码器的`synthesize_mel()`返回原生22.05k PCM，不能直接送入16k插槽；应使用`synthesize()`的16k结果，或提供明确且经过验证的16k适配层。

## 插入已有模块：推荐写法

将自己的真实模块对象包装成`CallablePCM16Stage`，然后按希望的顺序放入`stages`。没有自动发现、动态导入或隐式启用模块。

```python
from new25_postprocess import CallablePCM16Stage, OpusBWE24, RNDVoCWithBWE24

# real_module由调用方加载，必须是实际训练好的处理器。
# 下面假设它已有process_pcm16(pcm) -> (np.int16[N], dict)接口。
def process_extra(pcm, *, sample_rate):
    if sample_rate != 16000:
        raise ValueError("This module requires16k PCM")
    return real_module.process_pcm16(pcm)

stage = CallablePCM16Stage(
    name="my_waveform_module",
    process=process_extra,
    reset=real_module.reset,
    learned_parameters=None,  # 未核实时明确未知；实际部署再填写真实参数数目。
)
backend = RNDVoCWithBWE24(vocoder, OpusBWE24(), stages=[stage])
pcm24 = backend.synthesize(normalized, "zh")
```

如果模块本身无跨调用状态，或其整句处理方法已保证每次重置，可**明确**传`reset=None`。有循环、STFT或滤波缓存的模块应提供真实重置方法，不能用空函数掩盖状态未清空。多个模块用`stages=[stage_a, stage_b]`；按列表顺序串行处理，每句调用各自一次`reset()`。并发语句使用独立的backend和模块实例。

若自己的模型消费／输出的是浮点PCM，可在`process_extra`内部将PCM16转为`pcm.astype(np.float32)/32768.0`，运行真实模块，然后使用`new25_postprocess.bwe24.quantize`转换回PCM16。处理器需自行补偿padding、暖启动、flush及算法延迟，返回原长度；插槽不会自动拉伸、裁剪、补零或调整增益。

## 直接实现协议

也可实现[`PCM16Stage`](../new25_postprocess/chain.py)协议，直接放入列表，无需继承基类：

| 属性或方法 | 契约 |
|---|---|
| `name` | 非空字符串；同一链内唯一 |
| `input_sample_rate`、`output_sample_rate` | 都显式为`16000` |
| `learned_parameters` | 非负整数；未知时为`None` |
| `reset()` | 每个独立语句处理前调用；清空本模块状态 |
| `process_pcm16(pcm, *, sample_rate=16000)` | 输入一维`np.int16[N]`，返回`(np.int16[N], report)` |
| `report` | 有限、可JSON序列化的dict；记录模块来源、延迟补偿和必要参数 |

框架校验采样率、类型、维数和长度；给每个模块独立输入副本，并记录输入／输出SHA、顺序及重置信息。错误会停止链路，不自动旁路失败模块。长度相等只能证明接口长度，不能证明时间对齐、发音或音质，仍需用真实语音验证插入模块。

最终BWE消费最后一个模块的16k PCM16，输出长度为`(3*N+1)//2`个24k采样。`backend.last_report["intermediate_stages"]`保存每个中间阶段回执；没有中间模块时是空列表。

## 来源、精度与本次核验范围

- 原件来自Model提交`ad4799a75530903cd128e23e38c250602f7cea61`，路径`models/grouped11/fp32/shared/postprocess/opus_bwe`。原C权重和许可逐字节保留，来源清单在[`BWE_ASSET_MANIFEST.json`](../new25_postprocess/assets/BWE_ASSET_MANIFEST.json)。
- 原件有367,890个学习参数。加仓库同架构声学＋adapter＋RNDVoC后为4,170,882；额外插入模块的参数、缓冲和峰值RAM必须另计。10MB整个进程尚未验收。
- 此前真实模块验证在Linux x86_64用原CMake配置完成；小型数值／长度／状态重置检查通过，35k链路已生成75句450条三路对照。构建与试听回执见[`opus_bwe24_provenance.json`](../receipts/opus_bwe24_provenance.json)。
- 仓库声码器权重仍为明确发布的25k，35k只在既有模块试听证据中单列。本次不替换声码器权重、不提交音频、编译物、缓存、训练数据或运维脚本。
- 本次push只执行静态检查并复用既有运行证据；新增中间模块插槽没有在此次提交中运行真实外部模块。静态接口核对不称为新组合运行、音质或RAM验收。

BWE新增高频是估计；原35k未后处理的真实mel谱误差不能作为BWE24k或插入其他模块后的评分。第三方原许可保留在[`licenses`](../new25_postprocess/assets/opus_bwe/licenses/)和[`NOTICE.md`](../new25_postprocess/assets/opus_bwe/NOTICE.md)。
