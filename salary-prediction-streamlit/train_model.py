
# Salary Prediction Project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Salary_Data.csv")

print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe(include="all"))

# Cleaning
df = df.drop_duplicates()
df.columns = [c.strip() for c in df.columns]

# Fill missing values
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].fillna(df[c].mode()[0])
for c in df.select_dtypes(exclude="object").columns:
    df[c] = df[c].fillna(df[c].median())

# Feature engineering
bins=[0,2,5,10,100]
labels=["Entry","Junior","Mid","Senior"]
df["Experience Level"]=pd.cut(df["Years of Experience"],bins=bins,labels=labels)
df["Age_Experience_Ratio"]=df["Age"]/(df["Years of Experience"]+1)

# EDA
plt.figure(figsize=(6,4))
sns.histplot(df["Salary"],kde=True)
plt.tight_layout(); plt.savefig("salary_distribution.png"); plt.close()

plt.figure(figsize=(6,4))
sns.boxplot(x="Gender",y="Salary",data=df)
plt.tight_layout(); plt.savefig("gender_salary.png"); plt.close()

plt.figure(figsize=(8,4))
sns.barplot(x="Education Level",y="Salary",data=df,errorbar=None)
plt.xticks(rotation=30)
plt.tight_layout(); plt.savefig("education_salary.png"); plt.close()

plt.figure(figsize=(6,4))
sns.heatmap(df.select_dtypes(include=np.number).corr(),annot=True,cmap="Blues")
plt.tight_layout(); plt.savefig("correlation_heatmap.png"); plt.close()

X=df.drop("Salary",axis=1)
y=df["Salary"]

cat=X.select_dtypes(include="object").columns.tolist()
if "Experience Level" in X.columns:
    cat.append("Experience Level")
num=[c for c in X.columns if c not in cat]

pre=ColumnTransformer([
    ("num",Pipeline([("imp",SimpleImputer(strategy="median"))]),num),
    ("cat",Pipeline([
        ("imp",SimpleImputer(strategy="most_frequent")),
        ("oh",OneHotEncoder(handle_unknown="ignore"))
    ]),cat)
])

model=Pipeline([
    ("pre",pre),
    ("lr",LinearRegression())
])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model.fit(X_train,y_train)
pred=model.predict(X_test)

print("MAE:",mean_absolute_error(y_test,pred))
mse=mean_squared_error(y_test,pred)
print("MSE:",mse)
print("RMSE:",np.sqrt(mse))
print("R2:",r2_score(y_test,pred))

result=pd.DataFrame({"Actual":y_test.values,"Predicted":pred})
print(result.head())

sample=pd.DataFrame({
    "Age":[30],
    "Gender":["Male"],
    "Education Level":["Bachelor's"],
    "Job Title":["Software Engineer"],
    "Years of Experience":[5],
})
sample["Experience Level"]=pd.cut(sample["Years of Experience"],bins=[0,2,5,10,100],labels=["Entry","Junior","Mid","Senior"])
sample["Age_Experience_Ratio"]=sample["Age"]/(sample["Years of Experience"]+1)
print("Predicted Salary:",model.predict(sample)[0])


import joblib

joblib.dump(model, "salary_model.pkl")

print("Model saved successfully!")