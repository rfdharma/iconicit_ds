import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import plotly.express as px
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

st.set_option('deprecation.showPyplotGlobalUse', False)

X = pd.read_csv('Credit_card.csv').drop(columns=['Ind_ID'])
y = pd.read_csv('Credit_card_label.csv').drop(columns=['Ind_ID'])

X.bfill(inplace=True)

le = LabelEncoder()
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = le.fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42,stratify=y)

param_dist = {'colsample_bytree': 0.692708251269958, 'gamma': 0.15966252220214194, 'learning_rate': 0.0792681476866447, 'max_depth': 6, 'n_estimators': 466, 'reg_alpha': 0.6832635188254582, 'reg_lambda': 0.6099966577826209, 'subsample': 0.9165974558680822}


xgb_model = xgb.XGBClassifier(**param_dist)
xgb_model.fit(X_train, y_train)

def home():
    # Vertically align the content
    st.markdown(
        "<div style='display: flex; align-items: center; justify-content: center; flex-direction: column;'>"
        "<h1 style='text-align: center;'>⚜️ ICONICIT</h1>"
        "</div>",
        unsafe_allow_html=True
    )

    # st.image('kagurabachi.jpg')
    st.markdown('***')

    st.markdown(
        "<div style='display: flex; align-items: center; justify-content: center; flex-direction: column;'>"
        "<h5>R. Firdaus Dharmawan Akbar</h5>"
        "<h5>Dhia Alif Tajriyaani Azhar</h5>"
        "<h5>Ridho Pandhu Afrianto</h5>"
        "</div>",
        unsafe_allow_html=True
    )

def eda():
    st.title("📊 Exploratory Data Analysis")
    st.markdown("***")

    # st.text("")
    st.markdown(
        "<h5>The head of the data</h5>",  
        unsafe_allow_html=True
    )
    st.write(X.head(20))

def hypothesis_testing():
    st.title("🔎 Hypothesis Testing")
    st.markdown("***")

    # st.text("")
    st.markdown(
        "<h5>The head of pre-processed data </h5>",  
        unsafe_allow_html=True
    )

def modeling_page():
    st.header('Input Predict **Best Model (XGBoost)**')
    st.write('---')

def best_model():
    st.header('Information **Best Model (XGBoost)**')
    st.write('---')
    y_pred_test = xgb_model.predict(X_test)
    report_xgb_str = "XGBoost:\n" + classification_report(y_test, y_pred_test)
    
    st.write('**Detailed information performace best model (XGBoost) with 25% test data**')
    st.header('Confusion Matrix Best Model')
    col3, col4,col5 = st.columns([0.2,0.6,0.2])
    with col4:
        cm = confusion_matrix(y_test, y_pred_test)

        tick_labels = ['Normal', 'Tinggi']

        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=tick_labels, yticklabels=tick_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        st.pyplot()

    st.header('Metrics Evaluation Best Model')
    st.code(f"{report_xgb_str}", language='python')

    st.header('Interpretation/Explainable Best Model With SHAP Values')
    col1, col2 = st.columns([0.5,0.5])
    with col1:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, feature_names=X_train.columns)
        plt.title("Summary Plot")
        st.pyplot()

    with col2:
        shap.summary_plot(shap_values, X_test, feature_names=X_train.columns, plot_type='bar')
        plt.title("Summary Plot (Bar)")
        st.pyplot()


st.sidebar.title("Navigation")
selected_page = st.sidebar.radio(
    "Go to",
    ("Home", "Visual Analysis", "Hypothesis Testing", "Input Predict","Information Best Model")
)

if selected_page == "Home":
    home()
elif selected_page == "Visual Analysis":
    eda()
elif selected_page == "Hypothesis Testing":
    hypothesis_testing()
elif selected_page == "Input Predict":
    modeling_page()
elif selected_page == "Information Best Model":
    best_model()