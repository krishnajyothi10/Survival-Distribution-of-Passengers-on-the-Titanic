import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Medical Insurance Cost Prediction", page_icon="💡")
st.title("💡 Medical Insurance Cost Prediction")
st.image("https://insurance.phonepe.com/static/b9255dbb33d672b33828a697b2a55f45/836bc/basics_of_health_insurance_blog.webp", width=400)
st.write("Enter patient details below to predict insurance charges:")

# --- Paths ---
MODEL_PATH = Path(__file__).with_name("best_model.pkl")

# ⚠️ Set this to your training data path so we can fit the preprocessor if needed
DATA_PATH = Path(r"D:\PROJECT SUPERVISED MACHINE LEARNING\Worksheet in D  PROJECT SUPERVISED MACHINE LEARNING Medical Insurance cost prediction.xlsm")

@st.cache_resource
def load_or_wrap_model(model_path: Path, data_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path.resolve()}  (Python: {sys.executable})")
    obj = joblib.load(model_path)

    # If it's already a Pipeline, we're done.
    from sklearn.pipeline import Pipeline as _SkPipeline
    if isinstance(obj, _SkPipeline):
        pipe = obj
        outputs_log = getattr(pipe, "_outputs_log", True)  # default True
        return pipe, outputs_log

    # ---- Fallback: wrap a bare estimator with the SAME preprocessing learned from training data ----
    if not data_path.exists():
        raise FileNotFoundError(
            "Loaded a bare estimator (not a Pipeline) and cannot wrap it because the training "
            f"data file was not found:\n{data_path}\n\n"
            "Fix: Either save a full Pipeline as best_model.pkl, or update DATA_PATH to your training file."
        )

    df_train = pd.read_excel(data_path)
    required_cols = {"age","sex","bmi","children","smoker","region","charges"}
    missing = required_cols - set(c.strip().lower() for c in df_train.columns)
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")

    # Align column names
    df_train.columns = [c.strip().lower() for c in df_train.columns]
    X = df_train[["age","sex","bmi","children","smoker","region"]].copy()
    y = np.log1p(df_train["charges"])  # assume you trained on log target

    num_cols = ["age","bmi","children"]
    cat_cols = ["sex","smoker","region"]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
    ])

    # Fit preprocessor on training data (this recreates the original encoding & scaling)
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    pre.fit(Xtr, ytr)

    # Wrap the bare estimator into a pipeline
    pipe = Pipeline([("pre", pre), ("model", obj)])
    # mark that the target is log-transformed
    pipe._outputs_log = True

    return pipe, True

# --- Load prepared model (Pipeline or wrapped) ---
try:
    model, outputs_log = load_or_wrap_model(MODEL_PATH, DATA_PATH)
except Exception as e:
    st.error(f"❌ Failed to prepare model.\n\n{e}")
    st.stop()

age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
sex = st.selectbox("Sex", ["male", "female"]).strip().lower()
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
smoker = st.selectbox("Smoker", ["yes", "no"]).strip().lower()
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"]).strip().lower()

# allow manual override if you test a non-log model
override_log = st.checkbox("Model outputs log(charges)", value=outputs_log)

if st.button("Predict Insurance Cost"):
    X_raw = pd.DataFrame([{
        "age": int(age),
        "sex": sex,
        "bmi": float(bmi),
        "children": int(children),
        "smoker": smoker,
        "region": region,
    }])

    try:
        yhat = model.predict(X_raw)  # works for both native pipeline and wrapped model
    except Exception as e:
        st.error(f"❌ Prediction failed even after wrapping.\n\n{e}")
        st.stop()

    yhat = np.asarray(yhat, dtype=float).ravel()
    pred = np.expm1(yhat[0]) if override_log else yhat[0]
    st.success(f"✅ Estimated Insurance Charge: ${float(pred):,.2f}")
