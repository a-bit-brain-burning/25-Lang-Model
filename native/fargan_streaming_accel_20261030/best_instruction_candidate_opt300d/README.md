# Best instruction candidate: opt300d opt10（未最终化）

- 目标：在不改变数值结果的前提下削减 M55 已执行指令。基线 release O3 = 68,816,649 事件（344.083 ms @200 MHz 1 IPC）。
- 结果：opt10 = **62,678,844** 事件，**-8.919%**，1 IPC 乐观下界 **313.394 ms**；warm ROI（begin=`acoustic_begin`）= 60,661,130 → 303.306 ms。opt9 = 62,677,496（紧邻源码状态，差 0.002%），`results/` 记录 opt1/3/4/7/8/9/10 完整对比。
- 六项输出（condition/durations/output/output24/first24/fargan）与已认可 `accepted_i8_final` 基线**逐字节相同**；QEMU 通过、0 削波。ELF `6e4f58502ca958fd65f1a151b275b9ae9acc7cc9188d77188a3d0c731625f5c1`。
- `src/` 对应 `build/ordinary_o3_opt10/BUILD.json` 记录的源身份；相对 release 基线的改动清单见 `receipts/SOURCE_DELTA_VS_RELEASE_BASE.json`。
- **边界**：62.68M > 60M，300 ms 未达；未含完整原生文本前端/冷解压/外存等；本轮优化在 2026-10-30T23:24Z 后中断，无最终优化回执，**未采用为生产指针**。
- 目录：`src/ assets/ bwe_fast/ upstream/ third_party/`（构建输入）、`build_scripts/`（含原绝对路径的原始脚本）、`build/`、`reports/`（QEMU RESULT 与六输出）、`results/`、`profile/`（逐 PC CSV、GPL 插件源码）、`receipts/`（opt300c 发布 trace/会计回执与本候选 delta）。
