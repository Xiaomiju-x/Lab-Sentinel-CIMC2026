# AI-1 坩埚 4 类数据集采集指南 (CIMC Lab-Sentinel)

> 摄像头抓帧已打通(2026-05-29)。这份指南讲**怎么采数据 → 训练 → 部署**到 GD32。
> 训练在**本机 RTX 4050 笔记本**(模型很小,不需 5090;旧机是 RTX 4060)。

## 1. 四个类别(固定顺序,务必对齐文件夹名)

| idx | 文件夹名 | 中文 | 视觉特征(分类靠这些) |
|---|---|---|---|
| 0 | `empty` | 空坩埚 | 内腔暗、整体偏黑、无填充 |
| 1 | `loaded` | 装料未烧 | 白/灰**粉料**填充,颗粒纹理,亮 |
| 2 | `sintering` | 烧结中 | 炉内**红橙发光**,R 通道高,径向热核 |
| 3 | `done` | 出炉完成 | 烧结块,颜色变深、结块、低饱和灰褐 |

> 模型吃 **64×64 RGB**(保留颜色——发光是"烧结中"的最强特征,灰度会丢掉)。

## 2. 用哪个传感器:**OV5640,不用手机**

训练集必须和部署**同一颗传感器、同一几何**(同样的安装高度/角度/距离/光照)。手机色彩/FOV/畸变和 OV5640 差很远,拿手机训会域偏移掉点。手机只能当**补量 + 数据增强**辅助。

**安装**:把 OV5640 固定在最终监工位置,对准坩埚/炉口,采集时就是部署时的视角。

## 3. 怎么把帧弄到电脑

**方式 A(推荐,microSD 到货后)**:固件采集模式 → 每帧存 TF 卡 PNG → 拔卡拷到 PC。免线缆,适合炉边大量拍。**(SD 驱动 + 采集模式固件我在 SD 模块到货后写)**

**方式 B(UART-dump,现在就能用)**:固件经 UART 921600 把 QVGA 帧 dump 到 PC,Python 脚本收成 PNG。需笔记本接着板子。**(需要的话我现在就能加这个采集模式固件)**

## 4. 采多少 / 怎么变化

- 每类 **150~300 张**(共 600~1200)。小 CNN + 增强足够。
- **务必覆盖变化**:开/关灯、炉口余光、角度/距离抖动、不同坩埚、不同料、不同填充量。变化越多越鲁棒。
- `sintering`/`done` 这种台上烧不出的状态 → **去课题组真炉子拍**(他们就做这个,一下午能拍几百张)。

## 5. 目录结构(放好就能直接训)

```
CIMC/model/ai1_vision_cnn/data/crucible/
  ├── empty/      *.png|jpg
  ├── loaded/     *.png|jpg
  ├── sintering/  *.png|jpg
  └── done/       *.png|jpg
```
`train_crucible.py` 自动检测:有真实数据就用真实数据(否则退合成占位)。

## 6. ★ Claude API 自动打标(省人力)

采来的图先不用手标:写个脚本把每张图发 Claude vision API → 返回 `{class, confidence}` → 自动归入对应文件夹,你只复核**低置信度**的那几张。600~1200 张半小时搞定。
(这个标注脚本我可以写,等你确认 Claude API key 走哪个。)

## 7. 训练 → 部署(三条命令)

```bash
cd CIMC/model/ai1_vision_cnn
python train_crucible.py --epochs 40      # 1) 训练(自动用 data/crucible/),出 crucible_cnn.pt
python export_crucible_to_c.py            # 2) BN 折叠导出 → firmware/ai_models_c/ai1_crucible_weights.h + golden
cd ../host_test && clang ... crucible_test.c ai1_crucible.c ...   # 3) host golden 验证(已验:7e-7)
```
然后我把 `ai1_crucible_forward()` 接进 `vision_task`(QVGA→64×64×3 下采样 → 推理 → 喂 AI-4),Keil 编译烧录即可。

## 8. 现状(2026-05-29)

- ✅ 模型 `crucible_cnn.py`(64×64 RGB → 4 类,stride-2 conv,BN 可折叠,~12K 参数 ~3.9M MAC ~26ms@M7)
- ✅ 训练 `train_crucible.py`(本机 4060,合成数据验通 100%,真实数据即插即用)
- ✅ 导出 `export_crucible_to_c.py`(BN 折叠 vs torch 2.4e-7)
- ✅ C 引擎 `firmware/ai_models_c/ai1_crucible.c`(host golden **7e-7 PASS**)
- ⏳ 待:真实数据集 → 重训 → 接进 vision_task 烧录
- 当前权重是**合成占位**(键于亮度/发光/颜色,比 MNIST 强,但不是真坩埚);真实数据替换后就是真分类。
