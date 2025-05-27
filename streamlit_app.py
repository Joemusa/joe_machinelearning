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


st.title('🎈 Iris Machine Learning App')

st.write('This is builds a machine learning model!')
with st.expander('**Data**'):
  st.write('**Raw Data**')
  csv_url = ('https://raw.githubusercontent.com/Joemusa/joe_machinelearning/refs/heads/master/Iris.csv')
  df = pd.read_csv(csv_url)
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

  # Show in Streamlit
  st.pyplot(pairplot_fig.figure)







                       





