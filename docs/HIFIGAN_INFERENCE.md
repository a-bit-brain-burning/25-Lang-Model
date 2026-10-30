# 整句声学输出 → HiFi-GAN微调声码器

本入口复用同一套25语词表、语言ID和归一化文件，消费现有声学模型输出。它生成16 kHz mono WAV，不包含普通新文本到IPA/phonetic/context特征的完整前端。

## 安装

在仓库根目录安装带HiFi-GAN依赖的包：

```bash
python -m pip install -e '.[hifigan]'
python scripts/verify_release.py
python -m unittest discover -s tests -v
```

PyTorch和NumPy属于基础依赖；HiFi-GAN离线重采样另外需要SciPy。部署包不要求Matplotlib、librosa、训练优化器或训练语料。

## 命令行

先从兼容的原前端特征生成声学结果：

```bash
python -m new25_acoustic.inference \
  --features features.npz --language zh --model-dir models \
  --predict-durations --output acoustic.npz
```

`features.npz`包含`phones`、未归一化的`phonetic/context`及二值`pause`。也可沿用已有对齐时长，具体输入约定见仓库README。

然后把完整声学句子交给微调声码器：

```bash
python -m new25_hifigan.inference \
  --acoustic acoustic.npz --language zh \
  --model-dir models/hifigan --output output.wav
```

默认使用`vocoder_adapter_fp16.safetensors`，以FP32 CPU参考数学运行。选择`--weights vocoder_adapter_fp32.pt`会使用原FP32模型权重，结果可能与FP16舍入试听不同。写PCM16时明确限幅和量化；程序不会自动播放输出。

## Python接口

```python
import torch
from new25_acoustic import AcousticPredictor
from new25_acoustic.inference import LANGUAGES
from new25_hifigan import HiFiGANVocoder

acoustic = AcousticPredictor.from_pretrained('models', device='cpu')
result = acoustic.predict(
    phones, phonetic, context, 'zh', pause=pause,
)
vocoder = HiFiGANVocoder.from_pretrained('models/hifigan')
normalized = torch.from_numpy(result.normalized).unsqueeze(0)
language = torch.tensor([LANGUAGES.index(result.language)], dtype=torch.long)
pcm = vocoder.synthesize(normalized, language)
assert pcm.shape == (1, len(result.physical) * 160)
```

传入的是`result.normalized`，不是原始声学头的`prediction`，也不是直接未经归一化的`physical`。最后一维在`normalized`中已经由voicing logit转换为概率并按对应语言统计归一化，不要重复sigmoid。

## 整句边界

该实现按完整语句一次合成。`T`个声学帧输出恰好`T*160`个16 kHz采样点，不添加隐式时间偏移。中文使用31维，其他24语使用39维；语言ID沿用声学模型固定顺序。

本模型不接受跨调用的流状态，也没有保证`chunk1+chunk2`与整句推理相等。需要流式部署时，应另行处理上下文、重采样滤波状态、延迟和尾部flush，并做真实数值与音质验收。旧通用`VocoderStub`保留为未来流式实现的接口位置，当前HiFi-GAN使用上述独立整句API。

本入口的测试覆盖实现一致性、时长、维数、权重身份和有限值；它不证明新文本前端、音质/MOS或整个Linux进程10 MB目标通过。模型身份及已有开发指标见 [`HIFIGAN_MODEL_CARD.md`](HIFIGAN_MODEL_CARD.md)。
