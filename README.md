Phishing URL Classifier with Explainability

Interactive dashboard to detect phishing URLs and explain model decisions.
Features

Train Logistic Regression or Random Forest on Kaggle’s aman9d/phishing-data. (https://www.kaggle.com/datasets/aman9d/phishing-data)

Adjustable split/random state; shows precision/recall/F1, confusion matrix.

Feature insights: Pearson correlation & Mutual Information plots.

Per-URL explainability: SHAP waterfall for any picked URL.

Threat intel: VirusTotal API lookups (harmless/malicious stats, engine results).

“Pick 3 random URLs” mode to demo end-to-end.


Project structure
phishing-dashboard/
├─ app.py                         # Streamlit UI
├─ phishing_model/
│  ├─ train_model.py              # Training, saving predictions, metrics plots
│  └─ combined_dataset.csv        # (optional local copy; normally pulled via kagglehub)
├─ threat_intel/
│  └─ virustotal.py               # VT API helpers (no keys committed)
├─ outputs/
│  ├─ logreg_predictions.csv
│  └─ random_forest_predictions.csv
├─ requirements.txt
├─ .env.example                   # VT_API_KEY=your_key_here
└─ README.md
