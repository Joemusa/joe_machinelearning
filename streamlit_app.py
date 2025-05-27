# import plotly.express as px
import streamlit as st
import pandas as pd


st.title('🎈 Iris Machine Learning App')

st.write('This is builds a machine learning model!')
with st.expander('Data'):
  st.write('**Raw Data**')
  csv_url = ('https://raw.githubusercontent.com/Joemusa/joe_machinelearning/refs/heads/master/Iris.csv')
  df = pd.read_csv(csv_url)
  df
  
  features = df.columns[:-1]
  
  X = df[features]
  X
  y = df['Species']
  y
  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state = 0)
 






                       





