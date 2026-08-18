# CJK font public-source boundary

The competition build used a subset bitmap generated from a Windows SimHei
installation. Redistribution permission for that generated bitmap was not
established, so `lab_font_cn.c` is excluded.

Regenerate it with an OFL-licensed Noto Sans CJK / Source Han Sans font:

```bash
python -m pip install Pillow
python tools/generate_cjk_font.py \
  --font /path/to/NotoSansCJK-Regular.ttc \
  --vocab path/to/cluster_vocab.json \
  --output firmware/ai_models_c/lab_font_cn.c
```

Keep the font's OFL license and attribution beside your binary/release. The
generated file header records the input filename and SHA-256.

