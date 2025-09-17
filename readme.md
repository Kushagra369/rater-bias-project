# Rater Analytics — Demo

### Live Demo
- Streamlit Cloud: https://rater-bias-project-ytwmnhn55l73tuzbwcwsru.streamlit.app/

This project demonstrates how to measure inter-rater agreement (Cohen’s κ, Fleiss’ κ, Krippendorff’s α),
surface slice-level bias (chi-square), and compute throughput in a rater program.

Small project to demonstrate rater agreement metrics, bias slices, per-rater scorecards, and a minimal Streamlit demo.

Quickstart:
1. python3 -m venv .venv && source .venv/bin/activate
2. pip install -r requirements.txt
3. python analysis/bias_detection.py --input data/sim_rater_dataset.csv
4. streamlit run app.py
