import streamlit  as st
import pandas as pd
import joblib

# Load trained model
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "salary_model.pkl"

model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Salary Prediction App", page_icon="💼")

st.title("💼 Salary Prediction App")
st.write("Enter employee details below to predict salary.")

age = st.number_input("Age", min_value=18, max_value=70, value=30)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

education = st.selectbox(
    "Education Level",
    ["Bachelor's", "Master's", "PhD"]
)

job_title = st.text_input(
    "Job Title",
    value="Software Engineer"
)

experience = st.number_input(
    "Years of Experience",
    min_value=0,
    max_value=40,
    value=5
)

if st.button("Predict Salary"):
    if experience <= 2:
        level = "Entry"
    elif experience <= 5:
        level = "Junior"
    elif experience <= 10:
        level = "Mid"
    else:
        level = "Senior"

    ratio = age / (experience + 1)

    sample = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Education Level": [education],
        "Job Title": [job_title],
        "Years of Experience": [experience],
        "Experience Level": [level],
        "Age_Experience_Ratio": [ratio]
    })

    prediction = model.predict(sample)[0]

    st.success(f"Predicted Salary: ${prediction:,.2f}")