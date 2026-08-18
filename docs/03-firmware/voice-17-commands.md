# CI1302 fixed offline voice commands

CI1302 recognizes a fixed vocabulary locally and sends an 8-byte UART frame
with checksum. A bad checksum causes no action. This is not a free-form LLM.

| # | Chinese command | Intent |
|---:|---|---|
| 1 | 你好小亚 | wake |
| 2 | 开始烧结 | start controlled process |
| 3 | 结束烧结 | stop process and safe outputs |
| 4 | 暂停监测 | UI/TTS feedback; current process-state limitation applies |
| 5 | 继续监测 | UI/TTS feedback; current process-state limitation applies |
| 6 | 紧急停止 | unified safe stop and motor brake |
| 7 | 复位报警 | clear/recover according to deterministic gate |
| 8 | 查询温度 | navigate to Trend |
| 9 | 查询湿度 | navigate to Control; do not claim a spoken calibrated value |
| 10 | 查询气体 | navigate to Control; trend only, no ppm claim |
| 11 | 查询系统状态 | navigate to System |
| 12 | 打开风扇 | bounded dual-fan demonstration pulse |
| 13 | 关闭风扇 | fans off |
| 14 | 打开通风 | legacy ventilation-fan path |
| 15 | 关闭通风 | ventilation fan off |
| 16 | 测试灯 | compatibility UI/TTS path; visible legacy LED may be absent |
| 17 | 测试报警 | actuator test only after verifying the current relay polarity |

The table documents the defined vocabulary. It is not a blanket claim that all
17 commands were revalidated on every hardware/firmware revision. Public demos
should identify the actually prechecked subset.

