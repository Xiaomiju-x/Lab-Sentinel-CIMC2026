# 数据、许可与防泄漏台账

本目录只提供轻量、可审计的来源与准入材料，不复制数 GB 原始数据。

- `license_ledger.v1.json`：许可分类；
- `source_ledger.v1.json`：数据来源；
- `truth_ledger.v1.json`：真值类型与使用边界；
- `split_ledger.v1.json`：训练/验证/测试划分；
- `leakage_audit.v1.json`：防泄漏检查；
- `task_source_bindings.v1.json`：任务到来源的精确绑定；
- `exact_data_intake.v4.json` 及 `*_source_contract_audit.v1.json`：精确数据准入和逐源审计。

论文文本、教师/API 输出、fixture、模拟和结构派生数据均不冒充团队实验真值；`SIM_ONLY` 资产在 ModelBank 与回执中保持显式标记。

