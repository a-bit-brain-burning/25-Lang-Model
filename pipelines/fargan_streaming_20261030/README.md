# 主机 FARGAN 流式入口（原交付版）

四个入口脚本是交付时字节（`synthesize.py`、`runtime.py`、`acoustic_cursor.py`、`cea_lazy_condition_cursor.py`），已验证：20 ms 真实流包、Popen 首包在退出前可读、stdout 与 WAV 同包、stream/full 对照与取消/reset。

**原工作区布局要求**（脚本内保留原绝对/相对路径语义）：

```text
<work>/
  repo/                                  # 本仓库检出（提供 new25_acoustic/new25_fargan/models）
  new25_clarity_repair_20261030/
    repo/pipelines/prosody1000_rndvoc_opus24/   # 当前文本前端（GitHub 分支 fix/chinese-identifier-prosody-20261030，commit cea342e…）
  <本目录四个脚本>                        # ROOT=脚本所在目录；ROOT/repo 指向本仓库
```

默认 `--mode stream`；`--pcm-stdout` 输出 480 sample/20 ms PCM16@24k。`runtime.py` 会调用 `new25_fargan.inference_optim.freeze_weight_norm`（PCM 与未优化逐字节相同）。

回执：`receipts/STREAMING_COMPLETE.json`、`DELIVERY_VERIFIED_HOST_STREAM.json`、`SOURCE_IDENTITY_HOST_STREAM.json`、`ACOUSTIC_CURSOR_*`，以及最新 `cea342e` 同步的 `receipts/sync_cea342e_20261030/`（`SYNC_VERIFIED`/`PIPE_VERIFIED`/`FINAL_RESULTS` 等）。

主机计时不是 M55 周期；完整 25 语原生前端、双 10 MB 与 300 ms 未验。
