# RTX5090 实例迁移与断点恢复

本文件只适用于用户新提供的实例。任何旧 SSH 地址、账号或密钥都不得复用。

## 初次实例

1. 解压 `releases/forge200-gpu-ready-*.zip` 到同一工程根目录。
2. 复核包外 SHA-256，再逐项复核 `GPU_TRANSFER_MANIFEST.json`。
3. 建立新虚拟环境并安装 `pipeline/requirements-gpu-cu128.lock.txt`。
4. 分别执行 GPU-A/GPU-B `audit`，通过后才启动 30–60 分钟 pilot。
5. 产物固定写到 `artifacts/cloud5090/`；worker 每个任务更新 state、heartbeat 和 transfer manifest。

## 小时租期结束前

在实例正常关机/释放前，把以下内容整体复制回本地或用户控制的持久卷：

```bash
tar --zstd -cf forge200-cloud-checkpoints.tar.zst artifacts/cloud5090
sha256sum forge200-cloud-checkpoints.tar.zst > forge200-cloud-checkpoints.tar.zst.sha256
```

不得只复制 `best.pt`。必须同时保留 `last.pt`、worker state、每次 attempt log、promotion receipt、ONNX、golden、ICMF 包和 transfer manifest。

## 克隆或更换新实例

```bash
sha256sum -c forge200-cloud-checkpoints.tar.zst.sha256
tar --zstd -xf forge200-cloud-checkpoints.tar.zst
python pipeline/gpu_queue_worker.py --shard GPU_A --mode audit
python pipeline/gpu_queue_worker.py --shard GPU_B --mode audit
python pipeline/gpu_queue_worker.py --shard GPU_A --mode full --artifact-root artifacts/cloud5090 --resume
python pipeline/gpu_queue_worker.py --shard GPU_B --mode full --artifact-root artifacts/cloud5090 --resume
```

`--resume` 只跳过已有合法 promotion receipt 的完整任务；未完成任务从各 seed 的 `last.pt` 恢复。新实例仍须重新记录 GPU 名称、显存、PyTorch/CUDA 版本和 shard audit，不把克隆关系当作硬件身份延续。
