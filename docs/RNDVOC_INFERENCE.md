# RNDVoC25k接入与离线推理

安装仓库已有PyTorch/NumPy依赖及RNDVoC可选依赖：

```bash
python -m pip install -e '.[rndvoc]'
```

## 接入80k声学结果

复用[`new25_acoustic.inference`](../new25_acoustic/inference.py)生成的`AcousticResult` NPZ。输入契约与已发布HiFi-GAN相同：使用`normalized`波形条件，而不是原声学头logit或未经处理的`physical`。其中voicing已经过sigmoid，不要重复变换。25语的固定顺序、31/39维和归一化文件均与原声学权重绑定。

```bash
python -m new25_acoustic.inference \
  --features features.npz --language zh --model-dir models \
  --predict-durations --output acoustic.npz

python -m new25_rndvoc.inference \
  --acoustic acoustic.npz --language zh \
  --model-dir models/rndvoc --output output_rndvoc.wav
```

`features.npz`仍须由兼容的原文本前端提供；仓库尚未交付任意新文本的完整前端资产。

公开Python接口：

```python
from new25_rndvoc import RNDVoCVocoder

vocoder = RNDVoCVocoder.from_pretrained("models/rndvoc")
# normalized: CPU可读的[1,T,31]（zh）或[1,T,39]（其余语言）有限浮点数组。
pcm16k = vocoder.synthesize(normalized, "zh")
# 返回FP32 Tensor[1,T*160]，采样率16000Hz。
```

完整链：`normalized[1,T,31/39] → 原E2 adapter → raw logmel[1,80,M] → RNDVoC → 22.05k PCM → SciPy polyphase → 16k PCM`。adapter63个张量与已交付试听、仓库现有HiFi-GAN包逐字节相同，独立拆出保存，不需要加载HiFi-GAN Generator。

时间转换沿用原实现：

```text
L16 = T*160
L22 = ceil(L16*441/320)
M   = ceil(L22/256)
native waveform = M*256 samples
16k waveform = resample_poly(native,320,441)[:L16]
```

不添加旧80样本偏移，不作响度归一化、降噪、升频效果、去加重或教师预填。PCM16写入时才明确限幅和舍入。

## 直接80维mel接口

```python
native22k = vocoder.synthesize_mel(raw_logmel)
# raw_logmel: [1,80,M]；返回FP32[1,M*256]，采样率22050Hz。
```

输入必须已经逆归一化，使用自然对数，不能传z-score、dB或线性mel。固定契约是：22050Hz、80个Slaney mel频带、0–8000Hz、FFT1024、周期Hann1024、hop256、两端reflect384、`center=False`、幅度`sqrt(real²+imag²+1e-9)`、自然对数`log(clamp_min(1e-5))`。原始帧中心为`128+256*i`。

声码器采用与已有试听相同的显式OLA；不能直接换成`torch.istft(center=True)`，否则会移动半个hop并改变输出长度。投影缓冲保持原FP32值，不随可学习权重一起转成FP16。

## 精度、状态与验收范围

默认`vocoder_fp16.safetensors`是193个FP16可学习张量加三个FP32缓冲；加载后做FP32 CPU参考计算。可用`--weights vocoder_fp32.safetensors`切换未舍入Generator参考权重；adapter仍使用原FP16张量。权重加载先校验导出文件SHA，再核对名称、形状、精度和计数。

该实现为非因果整句离线推理。`state`或`chunk_frames`非空会报错；`synthesize_chunk`也明确拒绝调用。它没有实现[`VOCODER_INTERFACE.md`](VOCODER_INTERFACE.md)预留的逐块流式协议，不能将声学模型的分块状态直接传给它。

本次push仅执行序列化与静态核对，没有运行以上示例、模块前向、完整推理、训练、评估或性能实验。已有75句试听回执来自原检查点和原实现；包内新增接口的静态一致性不表述为发布入口运行验收。主观质量、任意文本、GPU性能和原生10MB进程边界见[`RNDVOC_MODEL_CARD.md`](RNDVOC_MODEL_CARD.md)。
