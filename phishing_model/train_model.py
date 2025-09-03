import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from threat_intel.virustotal import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import seaborn as sns
import kagglehub
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import shap
import time
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st

# initialize the necessary javascript libraries for interactive force plots
shap.initjs()

# reads data set
path = kagglehub.dataset_download("aman9d/phishing-data")
print(os.listdir(path))
csv_file = os.path.join(path, "combined_dataset.csv") 
url_df = pd.read_csv(csv_file)
feature_cols = [
    "ranking","isIp","valid","activeDuration","urlLen",
    "is@","isredirect","haveDash","domainLen","nosOfSubdomain"
    ]
X = url_df[feature_cols].apply(pd.to_numeric, errors="coerce")  # ensure numeric
y = url_df["label"].astype(int)

def render_metrics_dashboard(y_test, y_pred, title="Model performance"):
    # table like sklearn's text report
    rep = classification_report(y_test, y_pred, output_dict=True)
    rep_df = pd.DataFrame(rep).T.rename(columns={
        "precision": "Precision",
        "recall": "Recall",
        "f1-score": "F1",
        "support": "Support"
    })

    st.markdown(f"### {title}")
    st.dataframe(rep_df.round(3), use_container_width=True)

    # per class bars
    class_rows = [idx for idx in rep_df.index
                  if idx not in ("accuracy", "macro avg", "weighted avg")]
    chart_df = rep_df.loc[class_rows, ["Precision", "Recall", "F1"]]
    st.markdown("**Per-class Precision / Recall / F1**")
    st.bar_chart(chart_df)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=sorted({0, 1}))
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=sorted({0, 1}), yticklabels=sorted({0, 1}), ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    st.pyplot(fig, clear_figure=True)

    #KPIs
    st.markdown("**Summary KPIs**")
    c1, c2, c3, c4 = st.columns(4)
    acc = (y_test == y_pred).mean()
    c1.metric("Accuracy", f"{acc:.2%}")
    c2.metric("Macro F1", f"{rep_df.loc['macro avg','F1']:.2%}")
    c3.metric("Weighted F1", f"{rep_df.loc['weighted avg','F1']:.2%}")
    total = int(rep_df.loc["weighted avg","Support"])
    c4.metric("Support", f"{total:,}")

# traverse to dictionary with cvs files
def stats_urls(model_name):
    current_dir = os.path.dirname(__file__)
    outputs_dir = os.path.join(current_dir, '..', 'outputs')
    if model_name == 'log_reg':
       file_path = os.path.join(outputs_dir, 'logreg_predictions.csv')
    else:
       file_path = os.path.join(outputs_dir, 'random forest_predictions.csv')

    urls={}
    with open(file_path, 'r') as f:# pick 3 random URLs from model predictions and fetch their VirusTotal reports
        cvs = f.readlines()
        for i in range(3):
            random_index = random.randint(len(cvs))
            line = cvs[random_index]
            cut = line.index(',') 
            url = line[:cut]
            features  =line[cut+1:]

            urls[url] = {
              "features": features.rsplit(','),
               "virustotal_report": get_report(url)
           }
            time.sleep(1)  # to prevent overloading the API
    return urls

def save_results(model_name, model, X_test, y_test, y_pred):
    # build results with original domains + true label + features
    results = url_df.loc[X_test.index, ["domain", "label"] + feature_cols].copy()
    results.rename(columns={"label": "true"}, inplace=True)
    results["pred"] = y_pred

    # proba or decision margin
    if hasattr(model, "predict_proba"):
        results["score_phish"] = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        results["score_margin"] = model.decision_function(X_test)

    print(f"\n=== {model_name} ===")
    print(classification_report(y_test, y_pred))

    os.makedirs("outputs", exist_ok=True)
    out_path = f"outputs/{model_name}_predictions.csv"
    results.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

    return results
# trains data for log reg
def Logistic_Regression(test_size=0.3, random_state=42, return_model=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    save_results("logreg", clf, X_test, y_test, y_pred)
    explainer = shap.Explainer(clf, X_train, feature_names=X_train.columns)
    render_metrics_dashboard(y_test, y_pred)
    if return_model:
        return clf, X_train, X_test, explainer
 

# trains data for ran forest
def random_Forest(test_size=0.3, random_state=42, return_model=False):
    print('Random Forest model')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    save_results("random forest", rf, X_test, y_test, y_pred)
    explainer = shap.TreeExplainer(rf, X_train, feature_names=X_train.columns)
    render_metrics_dashboard(y_test, y_pred)
    if return_model:
        return rf, X_train, X_test, explainer


    



