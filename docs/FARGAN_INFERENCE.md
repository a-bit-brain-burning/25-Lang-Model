# FARGAN 4750 推理与连续流式接口

此入口使用固定窗口选中的4750步FARGAN和31/39维→20维adapter，直接消费现有`AcousticResult.normalized`。声学80k权重、音素词表和归一化统计复用`models/`原文件；`models/fargan/`仅含声码器和adapter的91个FP16张量。

```bash
python -m pip install -e '.[fargan]'
python scripts/verify_release.py
python scripts/verify_fargan.py
python -m unittest discover -s tests -p test_fargan.py -v

# features.npz需要兼容原前端的phones、phonetic、context及已有durations。
python -m new25_acoustic.inference \
  --features features.npz --language zh --model-dir models --output acoustic.npz
python -m new25_fargan.inference \
  --acoustic acoustic.npz --language zh --model-dir models/fargan \
  --seed 0 --output output.wav
```

输出`output.wav`为16kHz、mono、PCM16，另写`output.wav.json`说明语言、模型SHA、seed、输入/输出帧数及原时轴起止。浮点结果只在写WAV时限幅并四舍五入；sidecar记录超幅采样数量。`--metadata`可指定sidecar路径。所有输出请放仓库外。

## 时间对齐

输入是完整`[1,T,D]`归一化条件，`D=31`用于`zh`，其余24语`D=39`。需要`5≤T≤20000`。输出是`[1,(T−4)*160]`，对应未裁剪声学时间线的**`[400,160*T−240)`**。例如8帧输入给出640个采样，起点400、终点1040，原句总长1280采样。

这保留了选中模型原始有效中心、零teacher-prefill、零初始循环/激励状态和一次连续0.85去预加重。没有填充首尾、补零或重新对齐输出。要与参考波形比较，必须先按同一原时轴裁剪参考；不能将输出样本0直接对应原句样本0。

```python
from new25_acoustic.inference import AcousticPredictor
from new25_fargan import FARGANVocoder
from new25_fargan.inference import sample_layout

acoustic = AcousticPredictor.from_pretrained("models")
# phones/phonetic/context来自兼容原前端；durations为已有10ms帧时长。
result = acoustic.predict(phones, phonetic, context, "zh", durations=durations)
vocoder = FARGANVocoder.from_pretrained("models/fargan")
pcm = vocoder.synthesize(result.normalized[None], "zh", seed=0)
layout = sample_layout(len(result.normalized))
assert pcm.shape == (1, layout["output_samples"])
```

`language`也接受整数`[1]`tensor/array，顺序沿用现有25语合同。输入不能替换为原始声学头logit或未经归一化的`physical`。从NPZ读入时复用现有HiFi-GAN入口的声学NPZ校验与PCM16写入工具，不加载HiFi-GAN权重。

## 离线与复现边界

这是单句CPU、FP32参考计算，FP16仅用于参数存储；循环状态保持FP32。官方Python实现即使`eval()`仍会注入量化式噪声，接口用显式`seed`隔离随机数状态。相同平台、环境、输入和seed可重复；未认证跨PyTorch/CPU平台PCM逐字节一致。

运行时 `FARGANVocoder.streaming=True`。`begin_stream(language, seed=0)` 创建请求状态，
`synthesize_chunk(normalized_chunk, language, state=state, final=False)` 返回 `(pcm, state)`；
最后一块传 `final=True`，也支持空块 EOS。适配器历史、GRU、激励历史、随机数和去预加重状态均跨块保留。
`synthesize(..., chunk_frames=16)` 收集真实流式输出；原始整段参考保留为 `synthesize_offline(...)`。

```python
state = vocoder.begin_stream("zh", seed=0)
for start in range(0, len(result.normalized), 16):
    end = min(start + 16, len(result.normalized))
    pcm, state = vocoder.synthesize_chunk(
        result.normalized[None, start:end], "zh", state=state,
        final=end == len(result.normalized))
    # pcm is newly available audio; send it to the output before the next call.
```

不可在每个块上重建状态。总输出仍是原 `(T-4)*160` 有效中心，不是补首尾后的 `T*160` 协议。
发布资产 `model_config.json` 中历史 `streaming:false` 为原文件身份的一部分，保持 SHA 不变；
运行时能力由 `streaming=True` 和真实状态接口提供。流式与整段参考可能有浮点分块舍入差异，
不得称完整音频逐字节一致或人工音质已通过。

原声学入口可使用`--predict-durations`生成另一条推理路径，但4750模型的已有75句开发评分使用**真实对齐时长**，不能把该指标移植到预测时长或任意文本。普通新文本→IPA及原前端32/36维、22/34维特征的完整资产仍未交付。
