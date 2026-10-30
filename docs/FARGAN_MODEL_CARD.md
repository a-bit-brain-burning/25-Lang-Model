# 4750步FARGAN与物理条件adapter

用户指定上传的模型为4750步FARGAN，不是训练最后一步。原始检查点SHA256为`e90567f4f4ac8811d99b6cba910d117207a95a21be05cb5dbb0ebc1cfbc32e71`。原始推理包完整验收后拆分成仓库现有safetensors布局，不发布含声学权重的重复包或训练恢复状态。

| 组件 | 参数量 | FP16净字节 | 本次发布 |
|---|---:|---:|---|
| FARGAN Generator | 817,525 | 1,635,050 | `models/fargan/vocoder_adapter_fp16.safetensors`，28张量 |
| 31/39维→20维adapter | 1,891,412 | 3,782,824 | 同一文件，63张量 |
| 80k共享声学 | 1,186,071 | 2,372,142 | 复用已有`models/acoustic_fp16.safetensors`，97张量 |
| 单条完整FARGAN链 | **3,895,008** | **7,790,016** | 不含另一条HiFi-GAN链 |

新增safetensors文件含头部共5,426,602字节。FP16参数展开为FP32计算；进程、循环状态、临时数组、PyTorch和Python的占用均不包含在净权重字节中。**完整Linux推理进程10,000,000字节RAM尚未验收。**

## 选择依据

已有固定窗口开发实验在4000、4250、4500、4750、5000、5250六个检查点之间，按同75句、25语、冻结80k声学预测与真实时长→adapter→FARGAN的平均重建谱误差选择。封存test未参与。

4750平均谱误差`0.5400345599651337`，比4000降低1.4070%，比5000降低0.6297%，比5250降低0.0761%。只表明这六个候选在该开发指标上最低；不是人工MOS、音色、无噪声、统计显著或全局最佳认证。原指标摘要保存在[`fargan_development_summary.json`](../receipts/fargan_development_summary.json)，没有重新训练或扩大选择窗口。

## 工程身份与来源

[`models/fargan/EXPORT.json`](../models/fargan/EXPORT.json)绑定原4750检查点、完整来源包SHA、拆分权重SHA及已有声学依赖。91个新发布张量保留原FP16字节；原包中97个声学张量与仓库已有FP16声学状态逐张量、shape和dtype一致，词表和归一化整文件一致，因此本次不重复发布声学模型。

来源包此前已在独立Linux CPU进程从包内原始源码严格加载三组件，全部导出张量与选择评测权重逐值一致。本次仓库集成验证单列在[`fargan_release_validation.json`](../receipts/fargan_release_validation.json)；与原参考函数的两个维度短输入数值一致仅验证接口保真，不代替原75句开发评估。

FARGAN源自官方[Xiph Opus](https://github.com/xiph/opus)提交`bafd40dcbd506e12da257b8cb05c0afe1ed44719`及校验和绑定的官方模型归档。仓库保留所需Python源、[`COPYING`](../third_party/opus/COPYING)、[`LICENSE_PLEASE_READ.txt`](../third_party/opus/LICENSE_PLEASE_READ.txt)和精确来源SHA。`fargan.py`只将导入改为包内相对导入，`dnntools`初始化省去未用训练稀疏化导入；adapter源码按原字节保留。没有为微调权重或adapter额外授予许可证或商业权利。

## 功能和未完成验收

已提供兼容音素特征→已有声学API→归一化条件→FARGAN有效中心PCM/WAV入口；详见[`FARGAN_INFERENCE.md`](FARGAN_INFERENCE.md)。默认不修改原始边界或相位，CLI写入对齐sidecar。

完整任意新文本前端、完整原句首尾合成、连续分块协议、独立test、主观音质/音色/噪声、跨平台PCM逐字节一致和10MB进程RAM尚未认证。当前仓库不包含试听音频、partial归档、训练语料/特征、优化器、判别器、RNG、完整训练检查点或服务器运维记录。
