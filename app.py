
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from phishing_model.train_model import (
     url_df, X, y,
    Logistic_Regression, random_Forest, stats_urls,
)

# page setup
st.set_page_config(page_title="Phishing Dashboard",page_icon="🧊", layout="centered")
st.title("Phishing URL Classifier with explainability")

# app state (different windows)
if "stage" not in st.session_state:
    st.session_state.stage = "pick_model"  
if "model_name" not in st.session_state:
    st.session_state.model_name = None
if "trained" not in st.session_state:
    st.session_state.trained = False

# neatly presents API info
def vt_table(flat):
    stats = (flat.get("last_analysis_stats") or {})
    row = {
        "TLD": flat.get("tld", ""),
        "Reputation": flat.get("reputation", 0),
        "Times submitted": flat.get("times_submitted", 0),
        "Harmless": stats.get("harmless", 0),
        "Suspicious": stats.get("suspicious", 0),
        "Malicious": stats.get("malicious", 0),
        "Undetected": stats.get("undetected", 0),
    }
    return pd.DataFrame([row])

def plot_pearson_with_target(X_all, y_all):
    corr = X_all.corrwith(y_all)
    fig, ax = plt.subplots(figsize=(8, 4))
    corr.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Pearson correlation with target")
    ax.set_xlabel("Correlation")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig

def plot_mutual_info(X_all, y_all):
    mi = mutual_info_classif(X_all, y_all, discrete_features="auto", random_state=42)
    ser = pd.Series(mi, index=X_all.columns).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ser.plot(kind="barh", ax=ax)
    ax.set_title("Mutual Information (feature → target)")
    ax.set_xlabel("MI score")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig

# 1st window, pick model to train
if st.session_state.stage == "pick_model":
    st.write("Select a model to begin:")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Logistic Regression", use_container_width=True):
            st.session_state.model_name = "Logistic Regression"
            st.session_state.stage = "train_page"
    with c2:
        if st.button("Random Forest", use_container_width=True):
            st.session_state.model_name = "Random Forest"
            st.session_state.stage = "train_page"

# 2nd window, trains and displays results
elif st.session_state.stage == "train_page":
    model_name = st.session_state.model_name
    st.subheader(f"Model: {model_name}")

    c1, c2 = st.columns(2)
    with c1:
        test_size = st.slider("Test size", 0.1, 0.5, 0.3, 0.05)
    with c2:
        random_state = st.number_input("Random state", 0, 9999, 42, 1)

    if st.button("Train model", type="primary"):
        if model_name == "Logistic Regression":
            Logistic_Regression(test_size,random_state)  
        else:
            random_Forest(test_size,random_state)        
        st.session_state.trained = True
        st.success("Training complete.")

    # show correlation
    st.markdown("Feature to Target Relationships")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = plot_pearson_with_target(X, y)
        st.pyplot(fig1) # clear_figure=True
        st.caption("Pearson correlation with the target (linear association).")
    with col2:
        fig2 = plot_mutual_info(X, y)
        st.pyplot(fig2)
        st.caption("Mutual Information (captures non-linear dependence).")


    st.markdown("Explain 3 random URLs and VirusTotal info")

    if st.button("Explain now"):
        if not st.session_state.trained:
            st.warning("Please train the model first.") # prevents code from executing before training
        else:
            if model_name == "Logistic Regression":
                clf, X_train, X_test, explainer = Logistic_Regression(test_size,random_state,return_model=True)
                picked = stats_urls('log_reg') 
            else:
                clf, X_train, X_test, explainer = random_Forest(test_size,random_state,return_model=True)
                picked = stats_urls('rf') 

            virustotal_dict = {url: data["virustotal_report"] for url, data in picked.items()} # extracts an inner dictionary
            # loop through dict to show waterfall for each url
            for (url, vt) in virustotal_dict.items():
                st.markdown("URL under review")
                st.code(url, language="text")
                row = url_df.loc[url_df["domain"] == url, X_test.columns]
                row = row.iloc[[0]] 

                if (int(url_df.loc[url_df["domain"] == url, "label"].iloc[0])) == 0: # stores  actual label
                    check = "Confirmed non-phishing website"
                else:
                    check = "Confirmed phishing website"

                # shap
                exp_one = explainer(row)
                fig = plt.figure(figsize=(8, 5))
                if model_name == "Logistic Regression":
                     shap.plots.waterfall(exp_one[0], show=False, max_display=len(X_train.columns))
                else:
                     #break apart package so api can read it
                     vals = np.array(exp_one.values)
                     base = np.array(exp_one.base_values)
                     data = np.array(exp_one.data)
                     classes_ = getattr(clf, "classes_")
                     if check == "Confirmed phishing website":
                         pos_class = 1
                     else: 
                         pos_class = 0
                     class_idx = int(np.where(classes_ == pos_class)[0][0])  
                     if base.ndim == 1:         # (n_classes,)
                        base_scalar = float(base[class_idx])
                     elif base.ndim == 2:       # (1, n_classes) or (n_samples, n_classes)
                        base_scalar = float(base[0, class_idx])
                     else:                      # unexpected shape, flatten as fallback
                        base_scalar = float(np.ravel(base)[class_idx])
                     single = shap.Explanation(
                              values=vals[0, class_idx],          # (n_features,)
                              base_values=base_scalar,            # scalar
                              data=data[0],                       # (n_features,)
                              feature_names=exp_one.feature_names
                              )
                     shap.plots.waterfall(single, show=False, max_display=len(X_train.columns))
                st.pyplot(fig, clear_figure=True)
                st.caption(f"{check}")
                
                # API table
                st.markdown("**Additional threat intel (VirusTotal)**")
                st.table(vt_table(vt))
                st.markdown("---")
