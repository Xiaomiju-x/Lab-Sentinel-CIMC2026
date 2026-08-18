# GD32 Embedded AI Tool 代表工程链路

本 Git 目录以 AI-4 多传感器融合 MLP 为代表，保存公开可再分发的转换脚本与 host golden 测试源码；AI-2 自编码器是第二个转换样例。大模型输入、工具产物与经审核技术档案不放入普通 Git 历史，而随 `v1.0.1` Release 的技术证据包发布。

## Git 仓库内

- `conversion_scripts/`：导出、TFLite 解析、部署头生成及 INT8 对比脚本；
- `host_golden/`：测试源码及 2026-08-15 重新运行的标准输出。

板端 C 源位于仓库的 `firmware/ai_models_c/`，公开 Keil 工程位于
`firmware/keil_proj/project/`。

## GitHub Release 技术证据包

资产名：`lab-sentinel-cimc2026-technical-evidence-v1.0.1.zip`

SHA-256：`7d8610eb6728f9b9b6afd300ad98bd0c7f06c5975e05fd48ce8ac3d3d130f3d5`

包内 `03_gd32_embedded_ai_tool/` 另含：

- `input_models/`：AI-2、AI-4 的 `.pt`、`.onnx` 与配置；
- `tool_output/`：两个工具兼容 TFLite；
- `onchip_c_deployment/`：代表部署、自检和通用算子源码；
- `keil_r21_project_and_build/`：归档 Keil 工程和经过清理的构建/板测证据。

为避免泄露调试器唯一标识和发布无必要的二进制，公开技术包明确排除了 `.uvoptx`、`.axf`、`.hex` 与 `.map`。下载和双层 SHA-256 校验见 `docs/06-reproduction/host-smoke.md`。

## 验证结果

- `tflite_deploy_test`: 退出码 0，`TFLITE_DEPLOY ALL_PASS`，AI-4 最大绝对误差 `2.384e-07`；
- `aitest`: 退出码 0，20 个既有运行模型及 CAM/INT8/AdaptiveConformal/AI-4 工具部署检查均通过，`ALL_PASS=1`；
- 冻结竞赛构建记录：0 error，5 条既有 warning；这是历史回执，不是 GitHub CI 的 Keil 编译结果。

本目录是可复核的工具链入口，不包含 GD32 Embedded AI Tool 安装程序。官方工具产物与自研 C 内核在说明中严格区分，不把全部模型描述为官方工具自动生成，也不把 host golden 称为真板时延。
