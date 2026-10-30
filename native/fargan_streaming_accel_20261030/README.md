# M55 FARGAN 流式加速交接（2026-10-30）

固定现有 FARGAN 架构/权重（训练更新 0），M55/MVE 真实 QEMU 完整合成的两个版本：

| 目录 | 内容 | 状态 |
|---|---|---|
| `accepted_i8_final/` | 用户已认可量化音质的中文/英文完整固件、源码、资产、构建脚本、回执、试听 | 首音 QEMU 主机 2.2601 s（基线 3.6507，-38.09%）；300 ms/双 10 MB 未验 |
| `best_instruction_candidate_opt300d/` | 当前最快且六输出与已认可基线逐字节一致的指令优化候选（opt10）、源码、profile、回执 | 62,678,844 事件（-8.919%）；1 IPC 下界 313.394 ms；**300 ms 未达、未最终化、未采用** |
| `shared/` | i8 基线所需 BWE/上游 C 源码与 `arm-none-eabi-gcc` 包装脚本 | 工具链本体不入库（见下） |

## 构建说明

- 构建脚本是**原交付字节**，内部保留原工作区绝对路径（`/home/fomak/ffb/...`）作来源身份；重建需在同布局工作区，或按脚本中路径重定位后使用。
- 工具链：Ubuntu `arm-none-eabi-gcc 13.2.1 20231009`（`15:13.2.rel1-2`），`shared/bin/arm-none-eabi-gcc` 为原包装脚本，期望 `../toolchain/sysroot/...`；工具链二进制未随包（体积 1.4 GB）。
- 运行：`../../local_qemu_20261030/sysroot/usr/bin/qemu-system-arm -M mps3-an547 -cpu cortex-m55 ... -kernel firmware.elf`（QEMU 8.2.2，单 vCPU）。`reports/` 内 JSON 给出完整文件 SHA、逐项 checks 与 QEMU 主机时间。
- 动态指令计数使用随包 `profile/` 下 GPL-2.0-or-later QEMU 插件源码（`profile.c`、`qemu-plugin.h`，仅测量用，不链接进固件）。

## 主张边界

- QEMU 主机秒数不是 200 MHz 周期；`uncalibrated_1ipc_floor_ms` 是 1 IPC 乐观下界，不含冷解压、完整原生文本前端、外存/驱动。
- 完整系统 ROM+RAM 双 10 MB、原生 25 语前端、200 MHz 首音 300 ms、任意长文、人工音质全面验收均**未通过**。
- `opt300d` 为实验候选快照（优化轮未收尾），不替换默认生产指针或已交付数字。
