# Mid-fusion Case-level Demo (slim)

Browse-only Streamlit prototype. Risk/survival tables cover the frozen test set (**n=42**); dose slices + Mid Grad-CAM are packaged for **7 showcase cases**.
No online inference and no checkpoint loading.

## Layout

```text
.
├── app_mid_fusion_demo_tte_b.py
├── requirements.txt
├── README.md
└── demo_assets_b/
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_mid_fusion_demo_tte_b.py
```

## Streamlit Cloud

Main file path: `app_mid_fusion_demo_tte_b.py`

## Disclaimer

Research prototype only. Not for clinical decision-making.
