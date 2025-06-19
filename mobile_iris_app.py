import streamlit as st
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# -------------------- PAGE CONFIG -------------------- #
st.set_page_config(page_title="Iris Classifier 🌸", page_icon="🌼", layout="wide")

# -------------------- DARK BACKGROUND -------------------- #
def set_dark_theme():
    st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: #f1f1f1;
    }
    .css-1v0mbdj, .css-1y4p8pa {
        color: #ffffff !important;
    }
    .stSlider > div > div {
        background-color: #333 !important;
    }
    .stSelectbox > div > div {
        background-color: #333 !important;
        color: #fff !important;
    }
    .css-1cpxqw2 {
        color: #ccc !important;
    }
    </style>
    """, unsafe_allow_html=True)

set_dark_theme()

# -------------------- LOAD ANIMATION -------------------- #
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None

lottie_flower = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_kkflmtur.json")
st_lottie(lottie_flower, height=200)

# -------------------- LOAD DATA -------------------- #
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
feature_names = iris.feature_names

@st.cache_resource
def train_model(model_type):
    if model_type == "KNN":
        return KNeighborsClassifier().fit(X, y)
    elif model_type == "Decision Tree":
        return DecisionTreeClassifier().fit(X, y)
    else:
        return SVC(probability=True).fit(X, y)

# -------------------- ACCURACY LOGIC -------------------- #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
models = {
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(probability=True)
}
accuracies = {}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    accuracies[name] = round(accuracy_score(y_test, clf.predict(X_test)), 2)

# -------------------- RESPONSIVE INPUTS -------------------- #
st.title("🌸 Iris Flower Species Classifier")

device_type = st.radio("📱 Are you using a mobile device?", ["No", "Yes"], horizontal=True)

if device_type == "No":
    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    with col2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)
        model_choice = st.selectbox("Choose Model", ["KNN", "Decision Tree", "SVM"])
else:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)
    model_choice = st.selectbox("Choose Model", ["KNN", "Decision Tree", "SVM"])

st.markdown(f"📊 **{model_choice} Accuracy:** `{accuracies[model_choice]}`")

# -------------------- PREDICTION -------------------- #
model = train_model(model_choice)
sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("🌼 Predict"):
    pred = model.predict(sample)[0]
    probs = model.predict_proba(sample)[0]

    st.success(f"Predicted Species: **{target_names[pred].title()}**")
    st.info(f"Confidence: {round(probs[pred]*100, 2)}%")

    # Chart
    fig, ax = plt.subplots()
    ax.bar(target_names, probs, color=["#ff9999", "#66b3ff", "#99ff99"])
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction Probabilities")
    st.pyplot(fig, use_container_width=True)

    # Download result
    result_text = f"Prediction: {target_names[pred].title()}\nConfidence: {round(probs[pred]*100, 2)}%"
    st.download_button("📥 Download Result", result_text, file_name="prediction.txt")

# -------------------- DATASET TAB -------------------- #
with st.expander("📚 View Dataset"):
    df = pd.DataFrame(X, columns=feature_names)
    df["Species"] = [target_names[i] for i in y]
    st.dataframe(df)
    st.json(list(target_names))
