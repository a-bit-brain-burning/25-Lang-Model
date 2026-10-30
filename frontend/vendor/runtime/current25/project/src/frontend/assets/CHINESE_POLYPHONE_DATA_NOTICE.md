# 中文多音字资源来源与许可记录

本文件记录资源出处及本轮实际取得的许可文本，不对不同上游条款的适用范围作法律结论。

## 本地派生词表

`chinese_polyphone_phrases_v1.json` 中的 27,385 条新增词组，派生自
[`pypinyin-dict` 0.9.0](https://github.com/mozillazg/pypinyin-dict) 的
`phrase_pinyin_data.large_pinyin`。该包装项目引用
[`phrase-pinyin-data`](https://github.com/mozillazg/phrase-pinyin-data) 数据。

生成过程由 `project/src/experiments/chinese_polyphone_resources.py` 记录：仅保留纯汉字、长度 2–8、每字唯一读音、现有前端支持、原内置词表尚无且确实改变默认整词读音的词组；逐字核对现有单字多音候选，将“一／不”的词表读音规范为 `yi1`／`bu4`，将其他不在当前单字候选中的来源异读留在独立诊断文件中。没有按本轮探针答案手工挑词。

进入运行词表前的 compact v2 JSON SHA-256：
`80ff229fa75a7b14e71e8f390b4985c48667f60205da92744ff8840f06b0a856`。
它是词条原始紧凑 JSON 的散列；运行资产增加元数据包装后，文件散列可不同。
来源模块和派生产物的逐文件散列、过滤计数保存在
`project/outputs/chinese_polyphone_20261016/resources/compact_manifest.json`
与 `compact_v2_manifest.json`。

## 软件包装和上游数据分别保留来源

`pypinyin-dict` 的软件包装项目提供 MIT 许可。
`phrase-pinyin-data` 仓库也提供 MIT 许可文本；本目录保留其原文
[`PHRASE_PINYIN_DATA_LICENSE.txt`](PHRASE_PINYIN_DATA_LICENSE.txt)。

但 `large_pinyin` 是合并资源。按其
[官方来源说明](https://github.com/mozillazg/phrase-pinyin-data/blob/master/README.md)，
它合并 `zdic_cibs`、`zdic_cybs`、`cc_cedict`、`pinyin`、`di` 和 `overwrite`。
该说明还列出了汉典、字海、国学大师、漢語大詞典及早期拼音项目等参考来源。
本轮取得的生成模块不包含逐词来源映射。

相关数据来源：

- [phrase-pinyin-data 来源说明](https://github.com/mozillazg/phrase-pinyin-data/blob/master/README.md)
- [CC-CEDICT 项目](https://cc-cedict.org/)
- [MDBG 官方 CC-CEDICT 下载与许可说明](https://www.mdbg.net/chinese/dictionary?page=cedict)
- [汉典](https://www.zdic.net/)
- [字海](http://zisea.com/)
- [国学大师](http://www.guoxuedashi.com/)

本轮访问 MDBG 官方下载说明时，其当前 CC-CEDICT 数据标示
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)，并说明归属和相同方式共享要求。
这是当前官方页面的记录；`pypinyin-dict` 0.9.0 生成模块没有单独标出其所合并
CC-CEDICT 快照的日期、逐词出处及当时许可版本，因此本文件不将当前页面版本反推为所有历史数据的统一许可。
软件包装的 MIT 文本也不被用来替代上述第三方数据条款。

## g2pM 模型与本轮训练候选

本轮仅将 [kakaobrain/g2pm 官方项目](https://github.com/kakaobrain/g2pm)
作为独立研究对照。其官方代码许可原文保存在
[`G2PM_LICENSE.txt`](G2PM_LICENSE.txt)，为 Apache License 2.0。
该项目还使用 CC-CEDICT 派生的字音字典，出处与软件许可分别记录。

本轮实际取得的官方 `np_ckpt.pkl`、词表、类表及 `digest_cedict.pkl` 的 URL、大小、SHA-256，
保存在 `project/outputs/chinese_polyphone_20261016/resources/g2pm_official/download_manifest.json`。
该具体 checkpoint 实测为 401,548 参数，与官方 README 的概述数字不同。
原版没有作为默认运行模型；本轮压缩／训练候选另有 checkpoint、训练配置及来源记录，
不能把其评测结果归为未经改动的官方原版，也不据 Apache 软件许可推断其他来源数据的条款。

## 小型姓氏候选词表

另实际取得 [pinyin-pro 官方姓氏数据](https://github.com/zh-lx/pinyin-pro/blob/main/packages/pinyin-pro/lib/data/surname.ts)
及对应测试，原文件 SHA-256 为
`daff420d81a35e8abcc5b5295d265142a3c4380e89ccef843669a397669b7733`，
大小 9,828 字节。该项目的原 MIT 许可保存在
[`PINYIN_PRO_LICENSE.txt`](PINYIN_PRO_LICENSE.txt)。

实验目录 `project/outputs/chinese_polyphone_20261016/resources/consensus/surname_inspection/`
保留原源码、测试、来源散列及机械转换后的 `surname.json`。
转换保留全部 492 个姓氏（431 个单姓、61 个复姓），只将声调符号转为数字；
保留来源实际列出的读音候选，不从一般单字多音候选中擅自指定姓名读法。
当前来源每姓只列一个序列，这不被解释为现实中不存在其他姓氏读法。

官方的姓氏模式是显式姓名域选项，不能直接当作普通文章中的姓名识别器。
独立实验只核验了“姓＋完整最长姓氏＋真实边界”的 8 个正反例，未修改核心前端；
自动语法触发的实际接入及结果另记。原数据、转换后的资产和源码许可证分别保留出处。

## 训练支持的运行纠正子集

`chinese_polyphone_train_corrections_v1.json`通过全train统一门槛生成：8932句完整jieba词、至少3独立录音、citation一致>=.98，并与CC-CEDICT唯一citation一致。23个候选中的22项仅规范一／不，本轮运行子集机械排除这一类，留下1项词汇调纠正。记录数、支持数、原始SHA与拒绝证据保留于`resources/train_backed/`。这是BZNSYP既有训练数据派生资源；既有语料的使用限制见仓库ARTIFACTS，不借软件许可扩大数据授权。

## 运行姓氏资产

`chinese_polyphone_surnames_v1.json`与已保存pinyin-pro公开姓氏数据机械转换结果一致。运行时只在明确“姓＋完整姓氏”语境使用常用读法；不是通用人名识别，也不宣称来源列举现实中每个人的全部姓氏读法。显式用户读音优先。源资产中的NOT_RUN指转换阶段记录，主控运行集成与真实验收另见本轮manifest/report。
