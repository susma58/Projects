import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

df = pd.DataFrame({
    'Hours_studied': [5,3,8,2,7,6],
    'Sleep_hours': [6,5,8,4,7,6],
    'Passed': [1,0,1,0,1,1]
})
x = df[['Hours_studied', 'Sleep_hours']]
y = df['Passed']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.preprocessing import StandardScaler
df = pd.DataFrame({
    'Hours_studied': [5,3,8,2,7,6],
    'Sleep_hours': [6,5,8,4,7,6],
    'Gender': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male'], 
    'Passed': [1,0,1,0,1,1]
})
print(df)
df['Gender_Code'] = df['Gender'].astype('category').cat.codes
print(df[['Gender', 'Gender_Code']])
x = df[['Hours_studied', 'Sleep_hours', 'Gender_Code']]
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_scaled_df = pd.DataFrame(x_scaled, columns=x.columns)
print(x_scaled_df)
y = df['Passed']

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipe.fit(x_train, y_train)
pipe.predict(x_test)

import joblib
joblib.dump(pipe, 'model.pkl')
loaded_model = joblib.load('model.pkl')

import streamlit as st
model = joblib.load('model.pkl')
st.title("Student Pass Prediction")
hours = st.slider('Hours Studied, 0, 10')
sleep = st.slider('Hours Slept', 0, 10)
data = pd.DataFrame([[hours, sleep]], columns=['Hours_studied', 'Sleep_hours'])
result = model.predict(data)[0]
st.write('Prediction:', '✅ Pass' if result else '❌ Fail')

