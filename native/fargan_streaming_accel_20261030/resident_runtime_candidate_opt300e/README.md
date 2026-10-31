# Resident runtime candidate: opt300e opt16（完整 API 请求双口径 <60M 指令，1 IPC 口径）

- 目标：常驻固件完整 API 请求（`tts_begin_text_request` → 首个有效语音包）在保持输出逐字节相同的前提下低于 60,000,000 条已执行指令。
- **opt16_final** ELF `2118790e4ce4facc632c694344d41a493ae5f2f87f25bb063d666fb746c52535`（回退失败的 field-name-cache 实验后重建，ELF SHA 与 opt16_fast 完全一致）。
- 请求 ROI（begin=`acoustic_begin` → `snapshot_first_active_costs`，A0）：**59,747,483 事件 = 298.737 ms @200 MHz 1 IPC**，距 60M 余量 252,517。
- 完整请求 ROI（begin=`tts_begin_text_request`）：**59,988,134 事件 = 299.941 ms**，距 60M 余量 11,866（约 0.02%）；其中原生文本前端自身 240,651 事件。
- 本轮（opt16）改动：①boot 期整本中文词典页镜像（2439 页 / 1,516,448 B）解压进 SRAM `.zhprewarm`，请求内直接查表；②`zh_t2s` 首字节分派索引，去掉每位置全短语表扫描；③`bridge` state-only 参考流水线（不生成 scratch 字符串）＋只清零实际使用的 `N25TextFeatures` entry；④词典页二分改内联字节比较（去 `strcmp/strncmp/memchr`）；并沿用前几轮计数器中性化内核与 opt9–opt12 内核移植。
- 真实 QEMU：`NEW25_PERSISTENT_RUNTIME_PASS`，4 请求（A0/A1 完整、A2 70 包后 cancel、A3 完整），model init 1 次，**456/456 固件输出文件与 `final_resident` 基线 `run_001` 逐字节相同**；六项身份 condition `45a6073e…` / output `368f7b05…` / output24 `038e6545…` / first24 `5f4fb1d5…` / fargan `8519e0e5…` / durations `0054da57…` 不变。
- 同 ROI 进展（request / full）：常驻仪器化基线 81,071,945 / 71,366,967 → opt12 69,406,315 / 71,366,967 → opt13 59,943,780 / 61,904,328 → opt14 59,943,780 / 60,351,746 → opt15 59,747,454 / 60,155,449 → **opt16 59,747,483 / 59,988,134**。
- 段尺寸：`.text` 373,236 / `.assets` 6,229,728 / `.bss` 331,792 / `.stack` 131,072 / `.arena` 6,000,000 / `.zhprewarm` 1,516,448 B；DTCM bss end `0x300512d0` < stack bottom `0x30060000`。SRAM 词典镜像新增约 1.45 MB；总 RAM 约 7.9 MB，但**完整系统双 10 MB 必须随完整前端/平台重新验收**。
- **边界**：1 IPC 是 200 MHz 顺序核的最乐观计数线（MVE 64 位操作占两 beat、load/branch 额外周期，真实周期更高），余量仅 0.02%，且只覆盖固定普通中文句；**硬件 300 ms 未验收**，校准周期模型待做；完整原生 25 语前端、任意长文、完整系统双 10 MB、主观音质均未验。训练 0、权重/算术/默认指针不变。
- 失败实验保留：全局 field 名称缓存使 full 请求反而变慢（61.36M），已回退，未包含在 opt16_final 中。
- 目录：`firmware.elf`、`src/`、`frontend_source/`、`assets/`、`bwe_fast/`、`upstream/`、`third_party/`（构建输入）、`build_scripts/`（原字节脚本，保留原绝对路径）、`build/`（BUILD.json、sections/symbols）、`reports/`（QEMU RESULT 与六项 canonical 输出）、`profile/`（request 与 full-request 逐 PC CSV、GPL QEMU 插件源码）、`receipts/`（本轮 final_results）。
