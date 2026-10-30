# 当前完整文本 → 24 kHz WAV 交接版

本目录包含当前实际试听使用的完整链路与独立入口：

**文本前端 → 共享声学／韵律 → E2适配器 → RNDVoC25000 → Opus BWE24 → 24 kHz单声道PCM16 WAV。**

声学从所选2000步继续训练1000步，阶段累计1000/global3000，内置14,069参数韵律层。中文前端现采用 `best_bilingual_release_20261029` 的原生词典、注音、变调和词边界决策，并渲染成当前声学模型的数值特征；数字规范化沿用旧规则移植与前导零保护。另保留轻U=yiu1、长C、数字首位重读，以及长文本按自然句段处理的修复。25语使用同一声学核心；中文编号不切到英文声学，不逐字母合成WAV后拼接。追加3000步尚未执行，本目录权重不包含未来训练。

## 直接运行

在此目录或仓库根目录均可使用绝对/相对脚本路径。以下从仓库根目录执行，建议Linux x86_64、Python 3.10，已带该平台前端共享库和对应C源码。

```bash
python3.10 -m venv .venv-fullchain
source .venv-fullchain/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r pipelines/prosody1000_rndvoc_opus24/requirements-reference.txt
python pipelines/prosody1000_rndvoc_opus24/synthesize.py \
  --text '下一趟航班为CZ2671号，请前往T2航站楼。' \
  --language zh --output /tmp/flight24.wav
python pipelines/prosody1000_rndvoc_opus24/synthesize.py \
  --text 'The next flight is ready for boarding.' \
  --language en --output /tmp/english24.wav
```

每个WAV旁保存JSON，记录规范化文本、分段、预测时长、字母/首数字控制、Opus输入输出SHA和衔接。所有输出使用模型自行预测的时长。运行时无需旧工作目录、教师模型、训练语料、参考录音或额外下载模型。

Opus后端首次运行会用本目录自带的C源码和静态C权重在系统临时缓存编译。需C编译器（gcc或clang）；有CMake则走原CMake构建，无CMake时使用同源标量构建。后续复用缓存。没有启用GTCRN，没有Opus音频编解码。

前端四个共享库也可离线重建（需zlib、liblzma开发头文件）：

```bash
python pipelines/prosody1000_rndvoc_opus24/tools/build_frontend.py
```

该命令更新本地共享库和运行时库哈希；重建后的副本与原发布文件清单会有差异，应另存重建产物身份。

## 中文前端来源与适配边界

中文现在实际运行指定 D 盘发布包的 `zh_t2s` 注音规则：原词典、繁转简、多音字、轻声、一/不与三声变调、词分组及本调/表调信息。`reference_chinese/source` 包含完整依赖源码与原词典字节，`SOURCE_IDENTITY.json` 记录来源哈希。旧版 Kokoro token 与本模型词表不同，桥接层将语言学决策渲染为已训练的 initial/whole-rime、phonetic32/context34 特征；不更换声学或声码器权重，不取旧音频的时长。中文编号仍整段一次中文声学生成。

当前冻结词表缺少部分融合儿化韵母。遇到这些韵母时保留现有分读（如“这儿”），并在输出 JSON 的 `reference_frontend.deferred_fused_erhua` 中列出旧版目标读法与原因；不会把未知音素硬映射或称为与旧版完全一致。工作簿 49 条中文中 B31/B32/B37/B42/B47 存在此限制。裸“嗯”恢复旧版 en2；用户后来指定的轻 yiu1 U、长 C、首位数字强调保留。

原生词典每段最多128汉字，规划器会继续在自然句段内按112音素预算拆分，不截掉长文本。可离线重建（需 C 编译器、zlib 开发头文件）：

```bash
python pipelines/prosody1000_rndvoc_opus24/tools/build_reference_chinese.py
```

重建会更新该库的构建哈希。`provenance/CHINESE_REFERENCE_VERIFIED.json` 记录与原发布库的77段注音对照、数值特征契约及49条中文验证；试听和人工音质认可另行区分。该主机前端接入不代表新前端已经移植到 M55 或完整10MB系统RAM已验证。

## 灵活缩写与发音配置

前端先区分缩写类型，再交给已有G2P与模型：

| 输入 | 默认处理 |
|---|---|
| `CPU / api / PDF / SQL` | 逐字母读，英文U使用英文名称；中文逐字母U保留yiu一声 |
| `NASA / NATO / RAM` | 英文按单词读；中文恢复旧中英版的短全大写逐字母策略，可显式覆盖 |
| `U.S.B. / U S B` | 合并为一个逐字母缩写，长文不在内部切分 |
| `I'm / I've / NASA's` | 保持完整词形交给词典，不拆成单字母 |
| `Dr.Smith / e.g. / No.123` | 简写展开；规划句段时保护简写和后面的名称或编号 |
| `USB-C / GPT-4o / H.264 / HTTP/2` | 保持完整代码，保留字母与数字顺序及前导零 |
| `16 MB / 2 Mb / 24 kHz` | 先识别数值与单位，区分字节/比特，英文处理单复数 |
| `C++ / C# / .NET / Wi-Fi / I/O` | 使用配置中的明确读法 |

常见缩写和例外放在`frontend_profiles/default.json`。未知全大写词复用已有CMU词典：普通词走单词路径，字母串走逐字母路径；保留`I/A`在句子中的代词/冠词用法及`US/us`等大小写区别。自动规则有歧义时可以逐词指定。

```bash
# 强制将NASA逐字母读；航班号继续按完整代码处理。
python pipelines/prosody1000_rndvoc_opus24/synthesize.py \
  --text 'NASA uses a CPU.' --language en --abbreviation-mode spell --output /tmp/spell.wav

# 用示例配置让英文SQL读sequel，中文AI展开为人工智能。
python pipelines/prosody1000_rndvoc_opus24/synthesize.py \
  --text 'Open the SQL editor.' --language en \
  --frontend-config pipelines/prosody1000_rndvoc_opus24/frontend_profiles/example.json \
  --output /tmp/custom.wav
```

自定义文件按语言覆盖，未覆盖的默认规则继续生效：

```json
{
  "version": 1,
  "overrides": {
    "en": {"SQL": {"mode": "word", "spoken": "sequel"}},
    "zh": {"AI": {"mode": "expand", "spoken": "人工智能"}}
  }
}
```

支持`auto / spell / word`全局缩写策略，以及逐词`spell / word / expand`覆盖。逐词键区分大小写，优先于全局模式；`word`选择现有英语G2P路径，具体读法由词典决定，需指定sequel这类读法时填写`spoken`。`expand`可使用中文替换文本。中文词型缩写借用英语音素特征后仍由中文声学生成，不切换英语声学，不拼接字母录音。单独输入`USB`或`NASA`也可使用中文模式；仅借用静音上下文模板，模板汉字不进入合成音素。

每条JSON保留`reading_decisions`、规则来源、实际替换文本和配置SHA，便于定位读法。该更新仅调整前端；模型参数、声码器和训练步数不变。

```bash
python -m unittest discover -s pipelines/prosody1000_rndvoc_opus24/tests -p 'test_reading_policy.py' -v
python pipelines/prosody1000_rndvoc_opus24/verify_frontend.py --output /tmp/frontend-flex
```

## 自检

```bash
python pipelines/prosody1000_rndvoc_opus24/verify.py --output /tmp/fullchain-smoke
python pipelines/prosody1000_rndvoc_opus24/verify_manifest.py
```

自检覆盖全部25语短句、中文航班号/缩写/前导零、重复中文和中英文长文。`--strict-reference`会额外比对固定参考WAV的SHA；它适用于相同CPU参考依赖/数值环境，不承诺跨平台逐位相同。测试结果是工程验证，不能替代各语言人工音质、CER/WER或完整MCU验收。

## 文件与优化接口

|部分|文件|张量或接口|
|---|---|---|
|普通文本入口|`synthesize.py`, `text_worker.py`|文本/语言 → 句段特征 → WAV|
|中文规范化/缩写|`frontend_repair.py`, `frontend_live.py`, `legacy_contract.py`, `legacy_frontend/`|保持原文和规范化记录；迁移旧规则，实际参数仍为NEW25|
|25语词典/G2P/原生代码|`frontend_bundle/`|FEP词典、额外RO词典、Python/C源码、许可、主机共享库|
|声学与韵律结构|`acoustic_model.py`, `neural/new25_acoustic/`|8个编码卷积块、GRU、5个因果解码块、残差韵律层|
|声学存储与加载|`models/acoustic/`, `storage.py`, `loader.py`|136状态张量、320词表；矩阵INT8存储/CPU展开FP32|
|韵律与编号控制|`identifier_*.py`, `runtime.py`|预测时长/停顿/F0/能量，编号连读、轻U、长C、首数字重读|
|E2物理特征→mel|`models/rndvoc/adapter_fp16.safetensors`, `neural/new25_hifigan/model.py`|中文31维/其余39维 → 80维原始log-mel|
|RNDVoC|`models/rndvoc/`, `neural/new25_rndvoc/`|25000步；FP16学习参数＋FP32投影buffer，CPU参考运算|
|Opus BWE24|`neural/new25_postprocess/`|真实16→48k扩带，34个48k样点对齐，最终降采样至24k|
|完整清单与来源|`COMPONENTS.json`, `SHA256_MANIFEST.json`, `provenance/`, `licenses/`|文件/权重身份、上游许可与来源|

`new25_hifigan`在此仅提供已发布E2适配器和相同mel/重采样支持函数；当前链路不实例化HiFi-GAN Generator，也不加载其权重。仓库其他模型仍按原目录保留。

长文每个自然句段最多112音素，生成检查不超过1200个10ms帧，当前回归最大10.22秒。声学状态、RND全局统计和BWE状态分别按句段重置；只在自然文本边界处理静音边缘并衔接，航班号/时间/电话保持完整。整体为主机离线参考，不提供RND底层流式状态接口。

## 模型与验收范围

参数和文件身份见 `COMPONENTS.json`。`models/rndvoc/EXPORT.json`保留原发布来源，其中旧80k声学依赖属于历史说明；当前实际声学由本目录的`loader.py`和`models/acoustic/`明确绑定。声学1,637,644参数（含韵律层14,069），E2为1,902,992，RND为713,929，BWE为367,890，合计4,622,455可学习参数。RND另有345,249个FP32非学习buffer元素。提供的RND FP32备份是原25k未舍入对照，默认仍使用已试听的FP16参数版本。

本目录是完整主机文本到WAV交接，不是完整目标端固件或全INT8整数计算图。完整推理RAM≤10,000,000字节、M55持续实时播放和人工电音验收尚未通过，不能从存储文件大小推导RAM达标。当前试听与三条长文识别诊断也不能证明韵律层已全面学好。训练优化器/原始语料不属于此推理交接目录，原完整恢复点仍独立保留。
