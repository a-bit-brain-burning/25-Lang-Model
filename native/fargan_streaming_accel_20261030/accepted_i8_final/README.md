# Accepted i8_final（已认可量化音质的 M55 加速基线）

- 中文 ELF `87e096d0ecde9cdb56125ec4458055f4086d60339da1fd8938262000a0b3e2c1`，英文独立 ELF 见 `english_variant/`（`dd8370e2…`）。
- 同核顺序单次完整 QEMU：中文首有效 PCM24 `2.260072977 s`，对同主机核顺序基线 `3.650699934 s` 快 38.09%；整链 `21.569267013 s`（基线 32.3976 s）。英文首有效 `2.978162411 s`、整链 `15.655879007 s`。**QEMU 主机时间，非芯片周期。**
- 内存：后端 ROM 5,326,688 B；预留驻留 RAM 3,850,960 B；arena 峰 1,768,720 B；栈峰 109,568 B。不含完整原生前端/平台。
- 六输出与此前整数候选逐字节一致，0 削波；`reports/CANDIDATE_VERIFIED.json` 由实际文件核验生成。
- 用户已明确“量化之后的音质可以接受”（限交付试听）；仍**不是** 25 语/任意文本/人工全面质量验收。
- 结构：`src/`、`assets/`、`build_scripts/`（原字节，含原绝对路径）、`build/`（BUILD.json 与 sections/symbols）、`reports/`、`receipts/`、`listening/`（33 文件，含原 FP16 对照）、`english_variant/`（与中文共用 src，仅固定输入/weights.S 等不同）。
- 训练 0；`model.bin` 为合并 INT8 容器（与仓库原模型资产不同层级，未替换任何仓库权重）。
