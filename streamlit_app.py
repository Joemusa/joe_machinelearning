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
  df1 = df.drop(columns = ['Id'])
  
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
  X = df1[features]
  X
  y = df1['Species']
  y 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state = 0)
df1['Species'] = le.fit_transform(df1['Species'])
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
acc_score = round(knn.score(X_test, y_test)*100,1)

st.subheader("Model Description")
st.markdown(f"""
We have used the **K-Nearest Neighbors (KNN)** algorithm to build a predictive model based on the Iris dataset.
This model classifies iris flowers into one of three species by comparing the input flower's measurements to those in the training data.

The KNN model works by finding the 'k' closest data points to a new input and making predictions based on the majority class among those neighbors.
For this app, we selected **k = 1** as it offers a good balance between simplicity and performance.

After training the model and testing it on unseen data, we achieved an accuracy of approximately **{acc_score}%**.
This means the model correctly classifies new samples most of the time, giving us confidence in its predictions.
""")
st.sidebar('**Enter your measurements in cm**')


                       





