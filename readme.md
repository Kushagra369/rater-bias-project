# Rater Analytics — Demo

![CI](https://github.com/<your-username>/rater-bias-project/actions/workflows/ci.yml/badge.svg)

Inter-rater agreement (Cohen’s κ, Fleiss’ κ, Krippendorff’s α), bias slices (chi-square),
and throughput for rater programs. Built for interview walk-throughs.

## Live Demo
- Streamlit Cloud: https://<your-app>.streamlit.app

## Run locally
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_sim_data.py
python analysis/bias_detection.py --input data/sim_rater_dataset.csv
streamlit run app.py
