# Figure 7 · Mid-fusion 病例级演示（瘦身版 _b）

浏览型 Streamlit 原型。表格覆盖冻结测试集 **42 例**；剂量切片 + Mid Grad-CAM 仅 **精选 7 例**（float16 压缩）。  
无在线推理、不加载 checkpoint。整包约 **3 MB**，适合 GitHub 网页上传。

## 仓库结构

```text
.
├── app_mid_fusion_demo_tte_b.py
├── requirements.txt
├── README.md
└── demo_assets_b/
    ├── *.csv / *.json / *.parquet
    └── volumes/<case_id>/dose_cam.npz
```

## 本地运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_mid_fusion_demo_tte_b.py
```

## Streamlit Cloud

Main file path: `app_mid_fusion_demo_tte_b.py`

## 口径

Mid λ=0.15, c=0.201；主决策时点 24 个月。

## 免责声明

研究原型，不可用于临床决策。
