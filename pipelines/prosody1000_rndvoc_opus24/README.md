# 当前完整文本 → 24 kHz WAV 交接版

本目录包含当前实际试听使用的完整链路与独立入口：

**文本前端 → 共享声学／韵律 → E2适配器 → RNDVoC25000 → Opus BWE24 → 24 kHz单声道PCM16 WAV。**

声学从所选2000步继续训练1000步，阶段累计1000/global3000，内置14,069参数韵律层。前端包含旧中英 `flow_letters_v2` 的缩写识别/中文数字规范化迁移、轻U=yiu1、长C、数字首位重读，以及长文本按自然句段处理的修复。25语使用同一声学核心；中文编号不切到英文声学，不逐字母合成WAV后拼接。追加3000步尚未执行，本目录权重不包含未来训练。

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
