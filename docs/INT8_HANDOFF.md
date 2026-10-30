# 25 语完整 INT8 推理交接

本分支交付的是从普通文本前端到声码器的全部模型资产与可执行主机参考链。模型来自原仓库 `3a7465fc29e2ea11daed206dbf9401c8cfb18430`，沿用已完成试听的 `all_int8` 导出，没有新训练、重新量化或替换其他正在训练的模型。

## 直接运行

Linux x86_64 / Python 3.10+：

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-int8.txt
python scripts/verify_release.py
python scripts/verify_int8.py
python synthesize.py --language zh --text '你好，欢迎测试新的语音模型。' --output out/zh.wav --save-features out/zh.npz
python synthesize.py --language en --text 'Hello, this is a test of the new voice model.' --output out/en.wav
python scripts/smoke_text.py --output out/smoke
```

运行无需训练目录、旧 grouped11/M4 权重、教师音频、标注时长或额外字典下载。pip 安装阶段需要访问包源；依赖安装完成后推理可离线。25 语短句、NPZ 和 WAV 已在 `examples/host_smoke`，其 `RESULT.json` 记录零未知音素、有限输出和 WAV 哈希。它们是原创链路测试句，不是 CER/WER 测试集。当前模型质量未因这次交接获得重新验收。

库依赖为 glibc、libm、zlib、liblzma。非 Linux x86_64 环境需从随附源码重编译库；Ubuntu 构建依赖包是 `build-essential zlib1g-dev liblzma-dev`：

```bash
python scripts/build_frontend.py
python scripts/smoke_text.py --output out/rebuilt_frontend
```

重编译会更新 `frontend/vendor/frontend/libs` 和对应运行时哈希；因此发布总清单检查应在重编译前执行。可用 `--cc`、`--include-dir`、`--library-dir` 指定本地工具链及开发头文件。eSpeak NG 1.52.0 的匹配源码、音素模式补丁、Unicode 表和全部四个库的构建入口均随包提供。

## 资产与精度

| 部分 | 文件/目录 | 大小或约定 |
|---|---|---|
| 全链权重 | `int8/weights.safetensors` | 4,154,768 字节，316 原张量，4,015,048 参数 |
| 量化描述 | `int8/quantization.json` | 142 个矩阵/卷积权重 INT8，174 个偏置/归一化等保留 FP16 |
| 通道 scale | 同一 safetensors 中 `__scale__.*` | FP32，zero point=0 |
| 前端资源 | `frontend/vendor/frontend/packages/*.fep` | 16 份，共 3,875,677 字节 |
| 前端实现 | `frontend/api.py`、`frontend/vendor`、`frontend/sources` | 不加载旧网络权重 |
| 神经网络实现 | `int8/runtime/new25_acoustic`、`int8/runtime/new25_hifigan` | 精确对应当前包的完整 Python 模型 |
| 配置/统计/词表 | `int8/assets` | 包含语言顺序、归一化、285 音素词表及 HiFi-GAN 配置 |
| 原生权重接口 | `scripts/export_native.py`、`native/tensor.*` | 全量无损导出、逐张量 C 描述符与标量算子 |
| 双模拟器程序 | `simulator/prebuilt` | 可直接运行；完整权重加三个真实层测试 |

矩阵是**存储 INT8**。主机参考和随附 C 基线均以 FP32 激活/累加计算，不能把它们称为完整整数推理。线性层、Embedding、Conv1d 的量化轴为 0；`generator.ups.*.weight` 的布局为 `[Cin,Cout,K]`，量化轴为 **1**。scale 不能按普通卷积轴解释。非矩阵值不再量化。权重文件 SHA256：

```text
ba8824f6fe9cf07d2bedbfe981f723376d41b270aa8a99fec4f2573a08849bd6
```

模型文件加 FEP 资源为 **8,030,445 字节**；这不是固件大小。代码、解码缓冲、归一化/词表、内核和链接布局还需实际计入。历史 FP32/FP16、源码、许可证和双平台测试程序使 Git 仓库更大，仓库总大小也不是 ROM 大小。没有宣称完整 M4 固件 <10 MB、RAM 或实体硬件时延已通过。

## 推理图与移植约定

```text
UTF-8 text + language
  -> normalization / segmentation / pronunciation frontend
  -> phones[N], phonetic[N,32|36], context[N,22|34], pause[N]
  -> per-language feature normalization
  -> phone/language embedding + phonetic/context projections
  -> 6 noncausal LocalBlocks (width 128)
  -> prosody[5] + predicted phone durations
  -> frame layout + 4 position features
  -> frame projection + FP32 GRU (width 192, r/z/n gates)
  -> 3 causal LocalBlocks
  -> physical features[31 if zh, else 39]
  -> denormalize + sigmoid V/UV + vocoder conditioning normalization
  -> PhysicalMelAdapter (width 192, 6 blocks)
  -> interpolate to native mel frame grid
  -> HiFi-GAN (upsampling 8,8,2,2, folded weight normalization)
  -> 22,050 Hz waveform
  -> polyphase resample to exactly acoustic_frames*160 samples at 16 kHz
```

语言 ID 固定：`bg cs da de el en es et fi fr hr hu it lt lv nl no pl pt ro sk sl sv tr zh`。必须读取配置，不按字典迭代顺序猜 ID。前端原始数值需按 `int8/assets/normalization.json` 归一化；暂停标志来自音素，普通合成不使用参考时长。GRU 门排列、bias 位置和 FP32 状态按源码保留。声码器存在未来上下文，当前接口是完整句离线合成；把句子切块分别生成再拼接会改变结果。不要将旧 M4 声码器替换进来冒充当前网络。

逐层移植时先建立主机数值金标准：

```bash
python scripts/dump_trace.py --language en --text 'Hello.' --output out/trace_en
python scripts/dump_trace.py --language zh --text '你好。' --output out/trace_zh
# 如需逐帧 GRU 门线性层输出，可另加 --include-gru；输出文件会更多。
```

输出是无 pickle 的 `.npy`/`.npz`、模块调用顺序与 WAV，包含前端输入、声学条件、mel、各卷积层和原始声码器波形，便于定位首次偏差。只用于校准或数值对照，不据参考输入宣称普通 TTS 质量。

## 模拟器：立即跑通、再替换内核

已在 QEMU 8.2.2 的 `mps3-an547` / Cortex-M55 和 `virt` / RV32 上验证。已安装 QEMU 后，普通用户运行以下命令，不需要 root、编译器或 PyTorch：

```bash
python3 scripts/run_simulator.py --target m55
python3 scripts/run_simulator.py --target rv32
```

两者都应打印 `NEW25_REAL_INT8_KERNELS_PASS` 并退出 0。程序链接了全部 316 个原张量及 scale，实际计算并比较声学投影、梅尔适配器输入卷积、声码器首层反卷积三层的金标准结果；这不是完整文本到音频的目标端实现，也不是速度或峰值 RAM 验收。完整图调度、LayerNorm/激活/GRU、前端 C 调度、重采样与流式/内存优化仍需在目标端实现。

本次主机验证覆盖全部 316 张量逐值无损解码，以及 142 个真实权重矩阵/卷积算子；最大绝对差见 `receipts/int8_native_kernels.json`。优化可从替换 `native/tensor.c` 的标量函数开始，再用逐模块 trace 扩展到完整图。当前算子以 INT8 权重即时解量化到 FP32，不代表已实现 INT8 MAC/MVE 优化。

从源码生成和重编译：

```bash
python scripts/test_native.py --output build/native
python scripts/build_simulator.py --target m55 --assets build/native --output build/m55
python scripts/run_simulator.py --target m55 --elf build/m55/kernels.elf
python scripts/build_simulator.py --target rv32 --assets build/native --output build/rv32
python scripts/run_simulator.py --target rv32 --elf build/rv32/kernels.elf
```

交叉编译器分别为 `arm-none-eabi-gcc` 与 `riscv64-unknown-elf-gcc`，可通过 `--cc /path/to/compiler` 指定，不需全局安装。RV32 使用 `-march=rv32imafdc -mabi=ilp32d`；不要换成 RV64 ABI。M55 程序开启 FPU，代码在 `0x10000000`，数据/栈在 `0x30000000` DTCM，完整权重放在 `0x38000000` 的 8 MB SRAM secure alias，避免误塞进 512 KB ITCM。RV32 使用 virt 128 MB RAM、UART0 和 SiFive test 退出设备。此内存配置用于开发模拟，不代表最终硬件资源预算。

`export_native.py` 单独运行只需 numpy，可无损生成 `model.bin`（4,108,652 字节）、`model_index.c/h`、`MODEL.json`。`n25_tensor` 记录 offset、scale_offset、形状和轴，模型块是小端整数/IEEE754；导出后逐值恢复与原包完全一致。预编译文件包含相同块，SHA 与构建参数记录在 `simulator/prebuilt/MANIFEST.json`。

## 已知前端边界与验证范围

前端保留已有 FEP 压缩资源，PT/IT 采用当前模型训练音素表示。29 条既有输入的兼容检查：音素、音素特征、暂停数组全部相同；26/29 条上下文数组完全相同，BG 一句重音与 ZH 两句分词上下文有差异，见 `receipts/int8_frontend_compatibility.json`。因此不能声称新文本入口逐句复现原标注上下文或所有旧试听音频。

已有前端会明确拒绝无法映射的 IPA，例如丹麦语 `Hej` 的 eSpeak 输出含 `?ɑ`，当前规范音素表不支持；不会静默替换成任意音素。本次 25 语测试使用支持的短句验证完整依赖链，没有通过修改权重或伪造特征规避问题。合作者扩展规范化/发音支持时应保留旧用例的语义和回归对照。

主机库已从随附源码重建并完成全部 25 语文本到 WAV 测试。eSpeak 只做文本音素转换，音频合成入口按补丁明确禁用；音频来自本次神经网络。第三方许可、原有数据/词典归属分别保留在 `frontend/attribution`、`frontend/sources/espeak-ng/COPYING*` 及原仓库 `third_party`，不新增统一的模型许可。
