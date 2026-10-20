# Mid-fusion Case-level Demo (Grad-CAM hi-res)

Browse-only Streamlit prototype. Risk/survival tables cover the frozen test set (**n=42**);
dose slices + Grad-CAM are packaged for the **3 manuscript Grad-CAM cases** only
(Figure 5C, Figure S3C1, Figure S3C2). Panels are the exact high-resolution exports from
`figs_summary`. Interactive slices are bilinear-upsampled to 512 px.

No online inference and no checkpoint loading.

## Layout

```text
.
├── app_mid_fusion_demo_tte_c.py
├── requirements.txt
├── README.md
└── demo_assets_c/
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_mid_fusion_demo_tte_c.py
```

## Streamlit Cloud

Main file path: `app_mid_fusion_demo_tte_c.py`

## Disclaimer

Research prototype only. Not for clinical decision-making.
