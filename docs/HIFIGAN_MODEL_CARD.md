# HiFi-GAN VCTK V2 · 25语微调声码器

本版本是第二个有限声码器实验的真实最终结果：**31,403次Generator更新**，另有31,410次适配器更新和30,905次判别器更新。正式GPU阶段计时14,100.246531秒。发布只包含用于推理的Generator与PhysicalMelAdapter权重及代码，训练用判别器、优化器和语料不随模型交付。

## 模型身份和组成

最终检查点SHA-256：`c2a12e3ccd96486504d6c01d2b253e225f9e9b063a49339eed97e5217f02a803`。75句开发评测使用的就是此最终检查点，未另行声称它是所有中间版本中音质最好的候选。

| 组件 | 参数元素数 | FP16净权重字节 |
|---|---:|---:|
| HiFi-GAN Generator，weight norm已折叠 | 925,985 | 1,851,970 |
| 31/39维物理条件→80维mel适配器 | 1,902,992 | 3,805,984 |
| **本次追加的声码器部分** | **2,828,977** | **5,657,954** |
| 仓库现有80k声学模型 | 1,186,071 | 2,372,142 |
| **完整模型参数合计** | **4,015,048** | **8,030,096** |

FP16和FP32文件均是model-only权重；FP32文件不包含原优化器历史。FP16部署权重经过舍入，仓库CPU参考推理会展开为FP32计算。文件身份和逐张量来源以 [`models/hifigan/EXPORT.json`](../models/hifigan/EXPORT.json) 为准。

| 文件 | 用途 |
|---|---|
| `models/hifigan/vocoder_adapter_fp16.safetensors` | 219个Generator/适配器张量，FP16存储 |
| `models/hifigan/vocoder_adapter_fp32.pt` | 同一最终检查点的FP32纯张量状态 |
| `models/hifigan/config.json` | 绑定的官方Generator结构和mel配置 |
| `new25_hifigan/model.py` | 已折叠weight norm的Generator、残差块与PhysicalMelAdapter |
| `new25_hifigan/frontend.py` | 兼容mel和固定polyphase重采样数学 |
| `new25_hifigan/inference.py` | 整句离线推理、加载及WAV命令行入口 |

## 输入、精度与时间线

Generator原生输入为22,050 Hz语音对应的80维mel、hop256，输出22,050 Hz波形。mel使用FFT/window1024、Slaney 0–8000 Hz、幅度STFT和自然对数压缩；幅度下限为`1e-5`，不是对功率取对数。本项目的31/39维物理曲线通过独立训练的适配器转成此mel表示。

声学接口仍为16 kHz、160 samples/frame。给定`T`个声学帧，输出长度固定为`T*160`：先计算等效22,050 Hz长度、向上对齐至256采样的mel时间线，适配器按原帧中心位置插值，Generator合成后通过固定polyphase重采样回16 kHz并裁到该长度。使用与训练/开发评测一致的SciPy重采样默认参数，不添加预加重、去加重或历史80采样偏移。

本版本是**整句离线实现**。适配器、Generator和重采样使用上下文，不能把独立分块调用拼接后宣称与整句等价；它没有实现 [`VOCODER_INTERFACE.md`](VOCODER_INTERFACE.md) 所定义的连续流状态协议。完整句调用示例见 [`HIFIGAN_INFERENCE.md`](HIFIGAN_INFERENCE.md)。

## 已有真实开发结果

固定25语、每语3句，共75句。所有生成音频有限、最大削波比例为0。输入信息不同的三路结果分别为：

| 路径 | 平均谱误差↓ |
|---|---:|
| 冻结80k声学模型预测 + **真实时长** → 适配器 → 微调Generator | 0.5549892275 |
| 真实31/39维标签 → 适配器 → 微调Generator | 0.3694748271 |
| 从参考录音提取真实官方mel → 微调Generator | 0.3036957200 |

第三路使用了来自目标录音的mel信息，不能把它的指标赋给前两路或完整TTS。其均值也尚未达到原0.3门槛。该评测是已使用的开发集，没有MOS、音色保真、无噪声或独立测试通过结论。封存测试集未参与此模型的选择。

已有75句试听使用真实时长；仓库新增命令行也可消费声学模型自主预测时长的输出，但这不补齐普通新文本前端，也不自动构成该路径的完整质量验收。

## 内存与部署边界

4,015,048个FP16参数净大小为8,030,096字节，**这不是整个进程的RAM**。原实验Python/PyTorch参考进程在ZH/EN两句、batch1、FP16展开FP32时，实际峰值RSS为721,891,328字节。该次进程还包括官方工具间接引入的Matplotlib等依赖。

仓库的独立推理包只保留部署所需模块，移除了绘图和训练依赖。它的峰值RAM尚未重新验收，不能直接借用旧进程数值，更不能据权重大小宣称达到10,000,000字节目标。完整原生FP16引擎、任意新文本前端及整个进程的10 MB限制仍未通过验收。

## 来源和许可

官方上游为 [`jik876/hifi-gan`](https://github.com/jik876/hifi-gan/tree/4769534d45265d52a904b850da5a622601885777)，固定revision `4769534d45265d52a904b850da5a622601885777`。初始化来自官方README链接的VCTK V2权重，原SHA-256为`049ff085eeae70ff0e1add39cfd4caad0fcca4511d259a84f68a0d1c1bbb2659`。

原MIT许可与版权声明保留于 [`third_party/hifi_gan/LICENSE`](../third_party/hifi_gan/LICENSE)，来源及修改范围见 [`PROVENANCE.json`](../third_party/hifi_gan/PROVENANCE.json)。官方权重下载目录没有额外的checkpoint专用许可文件；此说明如实保留该来源事实，不为来源语料附加任何新的许可声明。
