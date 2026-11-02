# Best instruction candidate: opt300d opt12（未最终化，未作硬件验收）

- 目标：在不改变数值结果的前提下削减 M55 已执行指令。基线 release O3 = 68,816,649 事件（344.083 ms @200 MHz 1 IPC）。
- 本轮候选链（同一 ROI、每次真实 QEMU 完整合成、六输出与 integrated/final_native 中文参考逐字节相同）：
  `opt1 67,699,667 → opt3 67,162,694 → opt4 65,394,236 → opt7 63,469,996 → opt8 63,173,254 → opt9 62,677,496 → opt10 62,678,844 → opt11 62,222,710 → opt12 61,984,476`。
- **当前最佳 opt12**：冷 ROI `61,984,476` 事件，`-9.928%` vs release O3，1 IPC 乐观下界 **309.922 ms**；warm ROI（begin=`acoustic_begin`，排除每请求视图准备）`59,966,762` → **299.834 ms**，首次低于 60M 事件线（低 33,238 条）。
- ELF `03537bda40035d2a836e2a89e3cc27763cd8ea4863eb0caa9835b9224d15af69`；`.text` 213,672 B。六项输出（condition/durations/output/output24/first24/fargan）与已认可 `accepted_i8_final` 基线**逐字节相同**，QEMU 通过、0 削波；无权重/算术改动。
- `src/` 与 `bwe_fast/`、`upstream/` 对应 `build/BUILD.json` 记录的 opt12 源身份；相对 release 基线的改动清单见 `receipts/SOURCE_DELTA_VS_RELEASE_BASE.json`（19 个文件变更＋1 个新增 `src/mem_override.c`）。
- **关键限制**：warm 299.834 ms 只是把 profiling 起点锚点从 `tensor_views_prepare` 移到 `acoustic_begin`，**不是常驻固件实测**；常驻变体（A0/A1 两请求）尚未构建验证。Cortex-M55 为顺序核、MVE 64 位操作占两 beat，实际周期高于 1 IPC 乐观值，**不称硬件 300 ms 通过**；cold 61.98M 仍高于 60M 线。完整原生 25 语前端、双 10 MB、长文/人工音质仍未验。
- 目录：`src/ assets/ bwe_fast/ upstream/ third_party/`（构建输入）、`build_scripts/`（含原绝对路径的原始脚本）、`build/`（BUILD.json、sections/symbols）、`reports/`（QEMU RESULT 与六输出）、`results/`（opt1..opt12 全链）、`profile/`（opt12 cold/warm 逐 PC CSV、GPL 插件源码）、`receipts/`（round2 final_results、opt300c 发布 trace/会计回执与本候选 delta）。
- 优化轮在 2026-10-30T23:31Z 完成第二轮后停止，未生成下一轮回执；候选**未采用为生产指针**，opt10 的 ELF/证据仍在本分支上一提交 `09973ca` 的 git 历史中。
