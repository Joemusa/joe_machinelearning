# import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib 


st.title('Iris Species Clacifier ML App')

st.subheader('**Data Analysis**')
with st.expander('**Data**'):
  st.write('**Raw Data**')
  csv_url = ('https://raw.githubusercontent.com/Joemusa/joe_machinelearning/refs/heads/master/Iris.csv')
  df = pd.read_csv(csv_url)
  df
  
  df1 = df.drop(columns = ['Id'])
  features = df1.columns[:-1]
  X = df1[features]
  X
  y = df1['Species']
  y 
  
with st.expander('**Statistics**'):
  st.write('**Number of columns and rows**')
  show_d = df1.shape
  show_d
  st.write('**description of the data**')
  desc = df1.describe()
  desc

 
with st.expander('**Correlation Matrix**'):
  st.write('Correlation Table')
  corr = df1.drop(columns = 'Species').corr()
  corr
  fig, ax = plt.subplots(figsize=(10, 4))
  heatmap = sns.heatmap(corr, annot=True, ax=ax)
  st.write('Heatmap')
  # Display in Streamlit
  st.pyplot(fig)
  
  pairplot_fig = sns.pairplot(df1, vars=df1.columns[:-1], hue=df1.columns[-1])
  st.write('Feature Relationship By Species')
  # Show in Streamlit
  st.pyplot(pairplot_fig.figure)
  
features = df1.columns[:-1]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state = 0)
df1['Species'] = le.fit_transform(df1['Species'])
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
acc_score = round(knn.score(X_test, y_test)*100,1)


st.sidebar.title("Input Flower Measurements")

sepal_length = st.sidebar.number_input("Sepal length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.sidebar.number_input("Sepal width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.sidebar.number_input("Petal length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.sidebar.number_input("Petal width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# Main page info
st.title("Iris Flower Classifier")
st.write("This app uses a K-Nearest Neighbors (KNN) model to classify iris flower species based on measurements.")
st.write(f"Model Accuracy: **{acc_score}%**")

if st.sidebar.button("Predict Species"):
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=features)
    prediction = knn.predict(input_data)
    st.success(f"Predicted Species: **{prediction[0]}**")



