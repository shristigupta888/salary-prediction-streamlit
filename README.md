# 💼 Salary Prediction using Machine Learning

A Machine Learning web application built with **Python**, **Scikit-learn**, and **Streamlit** that predicts an employee's salary based on various input features such as age, education, experience, job title, and more.

---

## 📌 Features

- Predict employee salary instantly.
- Interactive and user-friendly Streamlit interface.
- Machine Learning model trained on salary dataset.
- Data preprocessing and feature encoding.
- Easy to deploy on Streamlit Community Cloud.

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib (optional)
- Seaborn (optional)

---

## 📂 Project Structure

```
salary-prediction/
│
├── app.py                  # Streamlit application
├── model.pkl               # Trained ML model
├── preprocessor.pkl        # Saved preprocessing pipeline
├── salary_data.csv         # Dataset
├── requirements.txt        # Required Python packages
├── README.md               # Project documentation
└── notebooks/              # Jupyter notebooks (optional)
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/shristigupta888/salary-prediction.git
```

### 2. Navigate to the project

```bash
cd salary-prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

After running the command, open the local URL displayed in the terminal (usually):

```
http://localhost:8501
```

---

## 📊 Machine Learning Workflow

1. Load the dataset.
2. Clean and preprocess data.
3. Encode categorical variables.
4. Split data into training and testing sets.
5. Train the regression model.
6. Save the trained model using Joblib.
7. Build the Streamlit web application.
8. Predict salary based on user inputs.

---

## 📷 Application Preview

You can add screenshots here.

```
assets/
    home.png
    prediction.png
```

Example:

```markdown
![Home Page](assets/home.png)
```

---

## 📦 Requirements

Install all required packages using:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`

```
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
```

---

## 🎯 Future Improvements

- Improve model accuracy.
- Add more input features.
- Deploy on AWS or Azure.
- Compare multiple ML algorithms.
- Add visualization dashboard.
- User authentication.

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Commit your changes.
5. Push the branch.
6. Open a Pull Request.

---

## 👩‍💻 Author

**Shristi Gupta**

- B.Tech CSE Student
- ITS Engineering College, Greater Noida

GitHub:
https://github.com/shristigupta888

LinkedIn:
(Add your LinkedIn profile link)

---

## ⭐ Support

If you found this project helpful, please give it a ⭐ on GitHub.

---

## 📄 License

This project is licensed under the MIT License.
