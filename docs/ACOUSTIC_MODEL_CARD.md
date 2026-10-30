# 25 语言声学模型说明

本发布包含已达到 **80,000 步训练上限**的声学模型，作用是把原前端生成的音素与特征转换为逐帧声学条件。发布不含声码器网络或权重，也不包含可移植的完整新文本前端。当前交付可用于声学推理、声码器研发和接口集成，不能据此宣称已完成任意新文本到音频的完整 TTS。

“80,000 步完成”描述训练进度。原训练回执的 `quality_pass=false`、`full_linux_ram_pass=false`、`smoke_test=false`，不表示音质、整套部署或内存预算已验收。

## 1. 发布组成与身份

| 项目 | 本版本 |
|---|---|
| 模型组件 | 共享声学核心与 31/39 维输出头 |
| 语言 | 25，固定顺序见下表及 `models/model_config.json` |
| 词表 | 285 项；`<pad>=0`、`<unk>=1`；按本次训练 IPA 字符串建立 |
| 声学状态项 | 97 个 tensor |
| 储存参数元素 | 1,186,071；该值是储存元素数，不是保证每个元素均被有效训练的另行统计 |
| FP16 tensor 数据 | 2,372,142 字节；包含格式头的文件为 2,380,398 字节 |
| FP16 主发布 | [`models/acoustic_fp16.safetensors`](../models/acoustic_fp16.safetensors) |
| FP32 对照权重 | [`models/acoustic_fp32.pt`](../models/acoustic_fp32.pt)，仅声学 state dict |
| 配套文件 | [`models/vocabulary.json`](../models/vocabulary.json)、[`models/normalization.json`](../models/normalization.json)、[`models/model_config.json`](../models/model_config.json) |
| 导出记录 | [`models/EXPORT.json`](../models/EXPORT.json) |
| 声码器交接 | [`new25_acoustic/vocoder.py`](../new25_acoustic/vocoder.py)、[`VOCODER_INTERFACE.md`](VOCODER_INTERFACE.md) |

本次导出从 SHA 已绑定的联合检查点提取 `acoustic.*` 项，并移除该前缀。联合容器处于 `vocoder_pretrain` 的第 60,000 步；这是容器所处阶段的计数，声学部分的训练进度为 80,000 步。不能把容器的 60,000 当成声学只训练了 60,000 步，也不能把两者相加。

独立声学检查点 SHA-256 为 `dbeb07dec11b3600a2f49c729353bc812b46aa9129032ada830a47927d38050f`；提取所用联合检查点 SHA-256 为 `a8219a415d3c2eb2421ebc3ad55fabfabda4c1826d0ff1c1f74a8d8059599d98`。已通过只读 CPU 加载原独立检查点，逐项确认 97 个声学 tensor 与发布的 FP32 文件完全相同，并确认 FP16 tensor 全部等于原权重的 FP16 舍入结果及已核对的预览导出。证据见 [`receipts/acoustic_provenance.json`](../receipts/acoustic_provenance.json)，其中 `tensor_state_sha256` 为 `2477b4d17f94fa95f61a9d276c147e05305d6acc06c517b093db34e62d87917a`。FP16 文件经过舍入，不应要求其浮点值与原 FP32 tensor 逐位相等。

| 发布文件 | SHA-256 |
|---|---|
| `models/acoustic_fp16.safetensors` | `9e9b17f22b3355e4b9a09b34fab5d022119a7964cd03cf702e8f1d68987bff02` |
| `models/acoustic_fp32.pt` | `c8c30641becc7de082ce192934761f1256007dbf4a3a786c1a0973ec31fc90c2` |
| `models/vocabulary.json` | `96193c14c4a7ec624c7a3857e9de9273de64ba6fceee14f430c186530dc3705f` |
| `models/normalization.json` | `b5db6854811557e26da1471ab22b1d8daed3dc63a22eb108d08cb13a9c4909b2` |

原训练完成回执的原样副本为 [`receipts/acoustic_completed.json`](../receipts/acoustic_completed.json)，SHA-256 为 `e0a10485ca8780445a4c83312cac75c6b3ac5d02d600df6310b446c8b0e69bd1`。完整训练检查点、优化器/RNG 状态、训练控制器、监控工程、语料、训练特征缓存、音频和凭据均不随发布交付。

## 2. 模型结构与计算精度

实现位于 [`new25_acoustic/model.py`](../new25_acoustic/model.py)：音素 embedding、25 语言 embedding、32/36 维 phonetic 和 22/34 维 context 独立投影到共享 128 维；六个非因果局部卷积块使用 dilation `(1,1,2,2,1,1)`；韵律分支输出每音素 5 项。按音素时长展开为帧后，192 维 GRU 与 dilation `(1,2,4)` 的三个因果局部卷积块生成 31/39 维声学头，并加入语言声学偏置。

GRU 使用 `r,z,n` 门顺序，门计算和持续状态为 FP32。FP16 safetensors 用于储存舍入后的权重；当前 [`AcousticPredictor`](../new25_acoustic/inference.py) 会将它们展开为 FP32 并执行 FP32 参考推理。FP32 对照文件保留提取的 FP32 声学权重。切换两份权重会改变数值，不能把“FP16 权重的 FP32 参考运行”说成“完整原生 FP16 部署”。

## 3. 输入契约

`AcousticPredictor.predict` 接受单个完整语句：

| 字段 | 形状与语义 |
|---|---|
| `phones` | 整数 `[N]`，本发布词表中的 ID；输入不含 padding ID 0，未知音素使用 ID 1 并应记录数量 |
| `phonetic` | 原前端的原始浮点 `[N,32]` 或 `[N,36]`；推理器按当前语言统计归一化 |
| `context` | 原前端的原始浮点 `[N,22]` 或 `[N,34]`；推理器按当前语言统计归一化 |
| `language` | 固定代码字符串，如 `zh`；其 embedding ID 由下表决定 |
| `durations`，可选 | 非负整数 `[N]`，单位为声学帧；允许零时长，和须为 1..20,000 帧 |
| `pause`，预测时长时必需 | 原前端给出的二值 `[N]`，`1` 表示 pause 类音素；源规则识别 `sil`、`sp`、`spn`、`<sil>`，不能由目标声学条件反推 |
| `chunk_frames`，可选 | 正整数；在完整句编码后切分声学帧解码 |

传入的是原始前端特征，不能提前归一化后再次交给 `predict`。不能重用来源不一致的各语言旧整数 phone ID，也不能裁切或补零使特征维数看似符合。所有输入须有限，phonetic/context 的实际宽度必须匹配本发布该语言的统计项。

| ID | 语言 | 声学 D | phonetic 宽度 | context 宽度 |
|---:|---|---:|---:|---:|
| 0 | bg | 39 | 32 | 34 |
| 1 | cs | 39 | 36 | 22 |
| 2 | da | 39 | 36 | 22 |
| 3 | de | 39 | 36 | 22 |
| 4 | el | 39 | 32 | 22 |
| 5 | en | 39 | 32 | 22 |
| 6 | es | 39 | 36 | 22 |
| 7 | et | 39 | 32 | 34 |
| 8 | fi | 39 | 32 | 34 |
| 9 | fr | 39 | 36 | 22 |
| 10 | hr | 39 | 32 | 34 |
| 11 | hu | 39 | 32 | 22 |
| 12 | it | 39 | 36 | 22 |
| 13 | lt | 39 | 32 | 34 |
| 14 | lv | 39 | 32 | 34 |
| 15 | nl | 39 | 36 | 22 |
| 16 | no | 39 | 36 | 22 |
| 17 | pl | 39 | 36 | 22 |
| 18 | pt | 39 | 36 | 22 |
| 19 | ro | 39 | 36 | 22 |
| 20 | sk | 39 | 36 | 22 |
| 21 | sl | 39 | 32 | 34 |
| 22 | sv | 39 | 36 | 22 |
| 23 | tr | 39 | 36 | 22 |
| 24 | zh | 31 | 32 | 34 |

词表和各类均值/标准差仅在训练划分拟合。原流程保留原验证划分并按来源组封存独立集合；这里未发布这些语料或其内容，也未宣称封存语料没有被其他历史模型见过。

## 4. 时长与帧布局

如果给出 `durations`，推理按给定时长展开；这可以用于排查声学误差，但不能作为“已通过自主预测时长”的证据。若没有给出时长，则根据韵律输出 `p[1,N,5]` 和原前端 `pause` 预测：

```python
duration_log1p = softplus(p[..., 0]) * (1 - pause) + softplus(p[..., 1]) * pause
duration = round(exp(duration_log1p) - 1).clamp(1, 400)
duration[(pause > 0) & (p[..., 2] < 0)] = 0
```

每个音素的正时长被限制在 1..400 帧；预测为不存在的 pause 可为 0。总帧数限制为 20,000，即 200 秒，这是推理保护边界，不是已经验收的最长语音能力。

`frame_phone[T]` 按时长重复音素索引；对时长 `n>0` 的音素，第 `j` 帧位置为 `[j/max(n-1,1), 1/n, j==0, j==n-1]`，共四维。零时长音素不产生帧。每帧 160 样本、16 kHz，因此时长为 `sum(durations)*0.01` 秒，不能混用毫秒和帧。

## 5. 输出与流式边界

`AcousticResult` 中 `.prediction`、`.physical` 和 `.normalized` 的形状均为 **`[T,D]`，不含 batch 维**，其中：

- `.prediction` 是声学头原输出；末维为 voicing logit，其他通道为归一化预测。
- `.physical` 是物理条件；末维已 sigmoid，其余通道已按语言统计反归一化。
- `.normalized` 是供声码器使用的条件；由 `.physical` 全维重新归一化，不能用 `.prediction` 替代。
- `.durations`、`.frame_phone`、`.position` 保存帧展开信息；`.acoustic_state` 保存最后一块的声学解码器状态。

物理通道定义、F0/能量语义、AP5 索引以及声码器交接见 [`VOCODER_INTERFACE.md`](VOCODER_INTERFACE.md)。该表示尚未确认可直接还原为任何通用 mel 表示。

声学解码器状态为 `gru[B,192]` 和三层 `conv` 历史 `[B,4,192]`、`[B,8,192]`、`[B,16,192]`；当前参考路径为 FP32。对同一句连续帧块需传回全部状态，不能只保存 GRU。新语句必须重置，不能把该状态传给声码器。

`predict(..., chunk_frames=...)` 每次仍先完整编码该句，然后在函数内部以 `state=None` 开始分块解码；它不是从上一轮 `predict` 续接新 token 的会话 API。返回最终状态用于检查/交接底层解码行为，当前 `predict` 没有外部初始状态参数。编码器使用左右上下文，不能把 token 分段后分别调用推理并宣称与完整语句等价。帧分块不消除完整音素编码、完整输出等存储需求。

## 6. 已验收与未验收的范围

| 事项 | 状态与含义 |
|---|---|
| 80,000 步声学训练进度 | 来源回执达到 `MAX_STEPS_REACHED`，`training_bound_only=true` |
| 声学组件独立导出 | 97 tensor、1,186,071 元素；文件哈希见 `models/EXPORT.json` |
| FP16 权重储存 | 已交付 FP16 safetensors；当前运行示例使用 FP32 参考计算 |
| 声码器 | 仅协议、输入/输出验证及抛错 stub；未交付可合成音频的实现 |
| 任意新文本前端 | 未交付完整原前端代码/字典；不支持据此直接宣称新文本 TTS 通过 |
| 音质 | `quality_pass=false`；无可报告的 MOS 结论 |
| 完整 Linux 单进程 RAM | 未通过验收；10,000,000 字节为整个进程预算，不能只计算权重 |
| 25 语言性能一致性、长句音质、实时性 | 尚需结合真实原前端和外部声码器另行测试 |

来源开发评估明确使用已有的原文本音素与特征，标记为 `original_text_derived_phone_features_NOT_new_text_frontend`。即使使用预测时长，也仍属于该输入模式。原前端的不同语言适配器/字典存在来源差异，不能换成一个默认 G2P/eSpeak 版本后宣称输入契约相同。

完整部署的资源测量应覆盖原文本前端、声学权重和计算、外部声码器、FP32 持续状态、张量缓冲与运行时开销。当前 Python/PyTorch 参考实现不构成 10 MB 部署样机。参数量、权重文件大小、CPU 能加载或简单数值检查通过，都不能替代整进程内存和听感验收。

## 7. 后续实现交接

下游应先复用本发布词表、固定语言映射和统计文件，对原前端的 IPA/phonetic/context 输出建立可复核的一致性样例，再接入满足协议的外部声码器。需要新文本能力时，须交付对应语言的前端实现、字典与版本来源，并验证不借助目标音频即可生成兼容输入。接入声码器后分别报告接口正确性、客观指标、听感评价和资源测量，保留各自限制。

本说明已核对的原始实现及哈希列于 [`VOCODER_INTERFACE.md` 第 8 节](VOCODER_INTERFACE.md#8-已核对的依据与易错点)。公开导出记录与逐项模型说明服务于复现声学接口，不包含原语料、音频、训练文本、连接地址或凭据。
