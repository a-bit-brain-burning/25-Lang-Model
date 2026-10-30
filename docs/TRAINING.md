# 声学训练复现

此目录保留已完成声学部分所用的模型、声学损失、训练数据读取、语言均衡采样和 batch 组装。`scripts/train_acoustic.py` 只构建独立 `Acoustic`。正式训练使用 CUDA；CPU 仅允许显式 `--smoke-test` 做至多两次 batch=1 的接口更新。

发布权重是 **model-only**。原声学 optimizer、GradScaler、随机数和采样器历史没有发布，因此不能从发布权重恢复原训练进度。`--warm-start` 会创建新 run、新 optimizer 和新的采样历史；`--resume` 仅接受本脚本在同一 run 中保存的完整训练 checkpoint。

## 数据准备

原始训练语料、音频、文本、对齐文件和 prepared cache 不随仓库发布。训练前需要自行取得符合原格式且有使用权限的语料。`scripts/prepare_corpus.py` 保留原准备算法，创建独立缓存，不修改原语料：

```bash
python scripts/prepare_corpus.py \
  --corpus /path/to/your/corpus \
  --out /path/to/new_prepared_cache \
  --reserve-test-seconds 300
```

输入根目录按 `languages/<language>/manifest.jsonl` 组织。25 个语言及顺序由 `new25_acoustic/contracts.py:LANGUAGES` 定义。每条原始记录需有原实现要求的 NPZ、strict overlay、PCM16、TextGrid 及相应 SHA-256、split/source_group/text 身份字段；脚本遇到缺失字段会失败。输入音频必须为 16 kHz、单声道、PCM16，样本数严格等于帧数 ×160。准备时只读取数据数组，禁止执行数据集 Python。

准备程序先冻结 split 身份，再从 train 的声学标签拟合每语言 normalization、从 train 的 IPA 建 vocabulary。组或文本相连的样本一起选入 sealed test；正式训练要求训练前已完成该划分。`heldout_for_this_run_available` 不代表测试语料在历史上从未被其他实验使用。

成功输出包含 `COMPLETE.json`、`manifest.jsonl`、`vocabulary.json`、`normalization.json`、`records/*.npz`、`SOURCE_MANIFESTS.json`、`development.json`、`sealed_test.jsonl` 和测试选择回执。训练读取 `manifest.jsonl` 中的 train；validation 只保留索引，sealed test 不进入训练列表。每语言都必须同时有 train 和 validation。

缓存 NPZ 的必要字段：

| 字段 | shape / 含义 |
| --- | --- |
| `phones` | `[P]`，train vocabulary 的整数 ID |
| `phonetic`, `context` | `[P,32/36]`、`[P,22/34]`，原前端特征 |
| `pause` | `[P]`，由 IPA 的暂停符号决定 |
| `durations` | `[P]`，非负整数帧数，和为 T |
| `curve` | `[T,31/39]`，物理声学曲线，末维为 voice 概率 |
| `f0_valid`, `voice_weight`, `silence` | `[T]`，严格帧标签/权重 |
| `phone_valid` | `[P]`，音素监督有效性 |
| `phone_prosody`, `phone_prosody_weight` | `[P,2]`，音素级 log-F0 / log-energy 和权重 |

声学训练只打开 prepared NPZ，不读取 PCM。`data.read_pcm` 留给准备程序验证原始音频时间线。

## 从头训练

安装依赖后，在仓库根目录执行：

```bash
python scripts/train_acoustic.py \
  --config configs/acoustic_train.json \
  --prepared /path/to/prepared_cache \
  --run /path/to/new_acoustic_run \
  --device cuda
```

`--run` 必须显式提供且是新的输出目录，不能位于冻结 prepared cache 内。默认参数来自原配置的声学阶段：seed `20261028`、batch size `4`、学习率 `2e-4`、AdamW betas `(0.8,0.99)`、weight decay `1e-5`、梯度裁剪 `5`，上限为 `80,000` 次成功更新或累计 `28,800` 秒声学阶段活动时间，以先达到者为准。CUDA 使用 FP16 autocast 和 GradScaler；参数/Adam master state、循环门计算及损失归约保留 FP32。连续 8 次 AMP overflow 会保留失败 checkpoint 并停止。

保留原全流程的外层预算上限：墙钟和累计活动时间各 `172,800` 秒，最多 `12` 次启动；声学阶段仍受上述 8 小时上限约束。续训不会重置这些预算。达到步数或时间上限仅表示运行边界，`quality_pass` 始终为 false，不能据此宣称听感、泛化或资源验收通过。

## 原采样、时间线和损失

`BalancedSchedule` 在 25 个语言之间逐 batch 轮转。每语言内部保留独立 shuffled epoch，耗尽后重新 shuffle；一个 batch 的所有 utterance 使用同一语言。每步使用完整 utterance，按最长音素数和帧数 padding，并用有效位置 mask 参与监督。真实 duration 展开 frame→phone 映射，同时产生 `[相对位置, 1/duration, 首帧, 末帧]` 四个位置特征；没有重排或静默移动时间线。

音素特征和声学曲线使用每语言 train-only 统计量。无效 F0 先以该语言均值填入再归一化，F0 loss 另受有效掩码限制；voice 标签是概率，输出头是 logit。phone-level log-F0 / log-energy 与对应帧级通道使用相同的 mean/std。

`new25_acoustic/losses.py` 保留原 `acoustic_loss` 的表达式和权重：

| 项 | 权重 | 原监督 |
| --- | --- | --- |
| `acoustic` | 1.0 | masked Smooth L1；排除 voice 通道，F0 仅有效帧 |
| `delta` | 0.25 | 相邻帧预测/目标差分的加权 L1 |
| `global_variance` | 0.10 | 通道 `2:d-6` 的 masked 时间标准差差异 |
| `voice` | 0.20 | voice logit 的带置信权重 BCE |
| `duration` | 0.30 | 常规/暂停两个 softplus 头，对 `log1p(duration)` 做 Smooth L1 |
| `pause` | 0.10 | 暂停音素是否存在时长的 BCE |
| `phone_prosody` | 0.15 | 带有效性/置信权重的音素 F0 与能量 L1 |

这是原声学数学的提取。独立模型省去了原组合模型中后续模块初始化对全局 RNG 的消耗，运行库、CUDA 内核和设备也可能影响数值轨迹；不承诺从头训练逐位重现历史 checkpoint 或同样质量。没有原 prepared 数据时，无法仅凭公开代码重建原训练样本或原 normalization。

## Model-only warm-start 与完整续训

使用发布权重初始化一个明确的新 run：

```bash
python scripts/train_acoustic.py \
  --prepared /path/to/prepared_cache \
  --run /path/to/new_warm_start_run \
  --warm-start models/acoustic_fp32.pt \
  --device cuda
```

脚本要求权重为无 `acoustic.` 前缀的纯 `Acoustic.state_dict`，并核对权重同目录下的 `vocabulary.json` 和 `normalization.json` 与 prepared 内容完全相同。ID 词表或 normalization 不匹配时不可直接套用该权重。warm-start 的步数从 0 开始；`ORIGIN.json` 和训练 checkpoint 会记录权重 SHA-256 和 `model_only_warm_start_new_run`。

本脚本保存的 run 可以完整续训：

```bash
python scripts/train_acoustic.py \
  --prepared /path/to/prepared_cache \
  --run /path/to/existing_acoustic_run \
  --device cuda \
  --resume
```

`latest.json` 绑定最新 checkpoint 的相对路径和 SHA-256。每次保存均包含 FP32 模型、optimizer、GradScaler、successful step、attempts、overflow 计数、Python/NumPy/PyTorch/CUDA RNG、采样器 shuffle/offset/cursor、逐语言 exposure/访问数和累计预算。源码、配置、数据回执或 smoke/formal 模式改变时会拒绝续训。完成步数上限的 run 不接受 resume；需要新 run 才能改变预算或配置。

发布模型用 `weights_only=True` 读取。自身训练 checkpoint 为了恢复 Python/NumPy RNG 使用完整对象加载，仅应恢复自己信任的本地 run。

## 停止、回执与 smoke

在 run 目录、其父目录或仓库根目录放置 `STOP`、`PAUSE` 或 `SAFETY_STOP` 可请求安全停止；SIGINT/SIGTERM 同样在下一步边界停止。停止标记会保留，存在标记时拒绝接续；脚本不会自动移除标记。checkpoint 与 `STOPPED.json` 保存后退出码为 `3`；达到更新次数上限退出码为 `0`。进程硬杀或硬件故障只能恢复最近已完成的 checkpoint。

run 中保存 `SOURCE_IDENTITY.json`（训练 Python 文件逐文件 SHA-256）、`PREPARED_RECEIPT.json`、`CONFIG.json`、`ORIGIN.json`、`budget.json` 和 `metrics.jsonl`，训练日志不复制语料文本。每 `1,000` 次成功更新保存 checkpoint，每 `25` 步更新 status，退出前也保存。剩余空间低于 4 GiB 会安全停止。运行锁阻止同一输出目录并发写入。

对自行建立的临时合成 cache 做 CPU 接口验证：

```bash
python scripts/train_acoustic.py \
  --prepared /path/to/synthetic_fixture \
  --run /path/to/new_smoke_run \
  --device cpu \
  --smoke-test
```

smoke 固定至多两次 batch=1 更新，允许没有正式 heldout 的合成回执，所有产物标记 `smoke_test=true`；上述 CPU 检查的 CUDA 更新数为 0。可用 `--smoke-stop-after 1` 验证停止路径，再用相同参数加 `--resume` 验证恢复。该检查不使用真实训练语料，也不证明质量或正式 CUDA 训练性能。
