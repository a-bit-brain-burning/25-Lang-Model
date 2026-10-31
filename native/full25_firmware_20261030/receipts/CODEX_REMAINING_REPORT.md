# codex 未完成内容：诊断与可行性结论（2026-10-31）

**未修改冻结物**：`build/native25_text_paths_v1/firmware.elf`（SHA 77390626…）保持不动；本轮只做定位、热点归因与候选方案实测估算，未重编、未改权重/算术/默认指针、训练 0。

## 1. codex 留下的真实状态

| 状态 | 语言 | 指令事件 |
|---|---|---|
| PASS ≤60M | bg cs da de el es et hr lv nl pl sk sl tr zh（15 语） | 47.4–55.0M |
| 略超 | fi 64.18M / hu 65.31M / no 65.04M / sv 62.03M | +2.0~5.3M |
| 重度超 | en 1279.95M / fr 900.15M / ro 658.35M / lt 578.18M / it 356.83M / pt 144.99M | 2.4x~21x |

冻结构建：ROM 9,885,272 B（余 114,728 B），RAM 物理预留 9,860,312 B（余 139,688 B）。

## 2. 重度超的根因（已定位到函数级）

- en 请求内 `lzma_decode` 666.5M 事件：对 `en-CMU.fep` 的 keys/0(284,949B) + keys/1(265,851B) + pool/0(271,442B) + ids(193,999B) 做**整段 LZMA 解码**，且按词重复约 20 次 → 每请求解码约 13MB 原始字节；
- 同时 `fep_keys_worker` 147.5M + `utf8` 139.5M + `strcmp` 49.3M 是按词线性扫描已解码 keys 段；
- fr/ro/lt/it/pt 的 lzma_decode 分别 439M/266M/254M/145M/58M，同一模式。

## 3. 四个候选修复方案的实测代价（均在本工作区算出）

| 方案 | ROM 增量 | RAM 增量 | 结论 |
|---|---|---|---|
| 词典转小页 DEFLATE（BG/SL 同款） | +2,468,387 B | 0 | 12.35MB ROM > 10MB，不可行 |
| 词典转小页 LZMA | +2,338,746 B | 0 | 12.22MB > 10MB，不可行 |
| 重分块 LZMA（chunk=64 词，含记录） | +2,286,630 B | 0 | 12.17MB > 10MB，不可行 |
| 请求内解码缓存（按语言驻留） | 0 | en 1.02MB / lt 1.57MB / fr 0.88MB / ro 0.42MB 等 | RAM 余量仅 139,688 B，不可行（arena 峰值 4,073,987/4,080,000 已满） |
| 换快速解压（LZ4/zstd 级） | 预计 +0.3~0.7MB | 0 | 仍超 10MB |

### 分块 LZMA 尺寸扫描（新增）

| 块大小 | 六语 ROM 增量 | 结论 |
|---|---|---|
| 256 词 | +1,494,513 B | 11.38MB 不可行 |
| 512 词 | +1,256,782 B | 11.14MB 不可行 |
| 1024 词 | +1,085,686 B | 10.97MB 不可行；且单词解码量随块增大而增加，与 60M 目标矛盾 |

## 4. 略超的 4 语

fi/hu/no/sv 超 2.0~5.3M（3.3%~8.8%），热点与 PASS 语言同分布（两个 packed_matrix kernel 占 41%），差异来自样例句更长（如 fi_00 有 107 units），属‘略超’而用户已表态可接受；~5–9% 内核微优化可争取压回 60M。

## 5. 需要用户决策

**A（最快）**：RAM 预算放宽 ~1.2MB（如 10MB→11.2MB 或按 128MB 目标板）→ 按请求解压并缓存词典 section，EN 可从 1280M 降到 ~100M 量级，再配合 key/index 优化力争 ≤60M；工程量 1–2 小时。
**B（保持双 <10MB）**：重构前端词典契约（句级预扫+小缓存/记录二进制重编码），数小时~数天，且需重验 125 核心输出逐字节一致。
**C（最小交付）**：冻结当前 native25_text_paths_v1：15 语 PASS、4 语略超、6 语列明待优化。

证据文件：`PAGED_LEXICON_ESTIMATE.json`、`PAGED_LEXICON_LZMA_ESTIMATE.json`、`CHUNKED_LEXICON_ESTIMATE.json`、`CODEX_REMAINING_ANALYSIS.json`；逐 PC 数据在 `profile/`，基线回执在 `reports/native25_text_paths_v1/RESULT.json`。
