# FARGAN 流式推理与 M55 原生加速交接（2026-10-30）

本分支在 `main` 原有离线 FARGAN 4750 接口之上，追加两条已完成/已验证的内容，并保持原 91 个 FARGAN 张量、声学、HiFi-GAN、RNDVoC、Opus BWE 权重与 `MANIFEST.json` 中未改动文件的字节身份不变；训练更新为 **0**。

## 一、主机 FARGAN 连续流式（`new25_fargan/` 与 `pipelines/fargan_streaming_20261030/`）

- `FARGANVocoder.streaming=True`：`begin_stream()` / `synthesize_chunk()` 保留 adapter 历史、3 个 GRU、FWConv、激励历史、独立 RNG 与 0.85 去预加重状态；`synthesize()` 为按真实流式收集整句的便捷入口，原整段参考保留为 `synthesize_offline()`。
- 输出仍为原有效中心 `(T-4)*160`，原时轴 `[400,160*T-240)`；不补首尾、不做 teacher prefill。发布资产 `models/fargan/model_config.json` 的历史 `streaming:false` 属文件身份，不修改。
- 第一项推理优化 `freeze_weight_norm` 只在内存中折叠 6 个重复 legacy WeightNorm hook，有效权重与优化前 PCM 逐字节相同，原 `n()` 噪声调用与 RNG 序列保留。
- 验证：`python -m unittest discover -s tests` 共 **29 项通过**（含新增流式状态/分块契约）；优化后 16k/24k 普通中文 WAV 与优化前逐字节相同；真实 Popen 20 ms 首包在进程退出前可读，stdout 与 WAV 同包。stream/offline 同下标 PCM16 最大 1 LSB、RMS 0.0623 LSB，已披露，不是逐字节相同或人工音质验收。
- 试听：`examples/fargan_streaming/listening/`（17 WAV，来自实际主机链，SHA 见其中 `MANIFEST.json`）。主机计时（已加载模型后首有效块约 76–86 ms、冷 CLI 首包约 3.28–3.37 s）不是 Cortex-M55/200 MHz 周期。
- 入口脚本按原交付版收录（绝对路径语义见该目录 README），最新一次 `cea342e` 前端同步回执在 `receipts/sync_cea342e_20261030/`；更新后的普通中文 WAV 与上一版同 SHA `37c4fa2c…`。

## 二、M55 原生加速（`native/fargan_streaming_accel_20261030/`）

1. **已认可音质的加速基线 `accepted_i8_final/`**
   - 中文 ELF `87e096d0ecde9cdb56125ec4458055f4086d60339da1fd8938262000a0b3e2c1`；英文独立 ELF `dd8370e2e95f5aa57f172cc82d86c4a77e0dc0e8f10edcc27d1ea6ac857c90e9`。
   - 同一主机核顺序、单次完整 QEMU：中文首有效 PCM24 主机时间 3.6507 s → **2.2601 s（-38.09%）**，整链 32.3976 s → 21.5693 s；英文首有效 2.9782 s、整链 15.6559 s。均为 **QEMU 主机时间，不是 200 MHz 周期或硬件实测**。
   - 后端 ROM 5,326,688 B、预留驻留 RAM 3,850,960 B、arena 峰 1,768,720 B、栈峰 109,568 B；不含完整原生前端/平台，不能当双 10 MB 达标。
   - 六项原始输出（condition/durations/output/first24/fargan 等）与此前整数候选逐字节相同，0 削波；用户已明确“量化之后的音质可以接受”（限该交付试听，不扩称 25 语/任意文本）。
   - 试听：`accepted_i8_final/listening/`（33 文件，含原 FP16 对照与 ASR/验证 JSON）。

2. **当前最快且输出逐字节一致的候选 `best_instruction_candidate_opt300d/`**
   - 在无报告 O2/O3 生产 trace 上继续做指令级削减；候选 opt1→opt12 的完整结果在 `results/`，最佳 **opt12** ELF `03537bda40035d2a836e2a89e3cc27763cd8ea4863eb0caa9835b9224d15af69`（opt11 = 62,222,710；opt12 = 61,984,476）。
   - 动态指令事件（同 ROI，QEMU 插件逐 PC 计数）：release O3 68,816,649 → **61,984,476（-9.928%）**；200 MHz 单发射 1 IPC 乐观下界 344.083 ms → **309.922 ms**；warm ROI（begin=`acoustic_begin`，排除每请求视图准备）59,966,762 → **299.834 ms**，首次低于 60M 事件线。warm 只是 profiling 起点锚点变化，常驻固件（A0/A1）尚未构建验证。
   - 六项输出与已认可 i8_final 基线 **逐字节相同**、QEMU 通过、0 削波；无权重/算术改动，两次独立 profile 结果一致（CSV SHA 记录在 `final_results/FINAL_RESULTS.json`）。
   - **未达标**：cold 61.98M 仍 > 60M；warm 59.97M 的 299.834 ms 只是移动 profiling 锚点的口径里程碑，不是常驻固件或校准后硬件时延；Cortex-M55 顺序核/MVE 两 beat 使真实周期高于 1 IPC 乐观值。该轮在 2026-10-30T23:31Z 完成第二轮后停止，**候选未采用为生产指针**，也未替换任何默认数值。

## 三、常驻固件请求口径（`resident_runtime_candidate_opt300e/`，opt16）

后续独立常驻谱系（opt300e_resident）把 opt9–opt12 内核移植进常驻固件，并把真实 API 请求路径的指令数压到 60M 以下：

| 口径（A0 首请求） | 事件数 | 200 MHz 1 IPC |
|---|---|---|
| request ROI（`acoustic_begin`→首有效包） | **59,747,483** | **298.737 ms** |
| full request ROI（`tts_begin_text_request`→首有效包） | **59,988,134** | **299.941 ms** |

- ELF `2118790e4ce4facc632c694344d41a493ae5f2f87f25bb063d666fb746c52535`；QEMU 4 请求（含 cancel）通过，**456/456 固件输出与 final_resident 基线逐字节相同**，六项身份不变。
- 同 ROI 进展（request/full）：81,071,945/71,366,967 → opt12 69,406,315 → opt13 59,943,780/61,904,328 → opt14 59,943,780/60,351,746 → opt15 59,747,454/60,155,449 → **opt16 59,747,483/59,988,134**。
- 主要手段：boot 期整本中文词典 SRAM 预热（2439 页/1,516,448 B，新增 `.zhprewarm`）、t2s 首字节分派索引、state-only bridge 参考流水线、定向清零、内联词典二分；并沿用计数器中性化内核。
- **边界**：1 IPC 是顺序核最乐观计数线（MVE 两 beat / load-branch 附加周期使真实周期更高），2 个口径余量分别 252,517 与 **11,866（0.02%）**，仅固定普通中文句；**硬件 300 ms 未验收**，校准周期模型待做。新增 SRAM 词典镜像 1.45 MB，完整系统双 10 MB 需随完整前端重新验收；完整原生 25 语前端、任意长文、主观音质未验。训练 0、权重/算术/默认指针不变。

## 四、边界

- 训练更新 0，权重/架构/声码器均未换；原主机 FP16/effective-F32 与原生 INT8 数值不同，不能互称逐字节一致。
- 完整原生 25 语文本前端、完整系统 ROM/RAM 双 10 MB、200 MHz 首音 300 ms、任意长文与人工音质验收仍未通过；常驻 opt16 的 60M 以下只是 1 IPC 计数线，不是硬件时延。
- 精确 SHA、逐文件回执、QEMU 报告与动态 profile CSV 均在两个目录的 `receipts/`、`results/`、`profile/`、`reports/` 下。
