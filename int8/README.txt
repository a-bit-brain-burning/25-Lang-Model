25-Lang-Model 压缩推理候选：all_int8

源码版本：3a7465fc29e2ea11daed206dbf9401c8cfb18430
原声学、HiFi-GAN和适配器结构不变；矩阵按输出通道对称INT8，其他值保留原FP16。
加载时反量化为FP32，CPU推理；不是已验证的原生INT8固件或10MB RAM实现。
自带匹配代码、词表、归一化和配置；完整新文本前端仍未交付。
使用已有兼容特征：
python synthesize_features.py --features /path/to/features.npz --language zh --output /tmp/out.wav
features.npz需要phones、phonetic、context、pause；时长由模型预测，不读取标注时长。
音频质量等待用户试听；原权重、旧认可模型及其他训练任务未更改。
