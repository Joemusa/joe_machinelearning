# import plotly.express as px
import streamlit as st
import pandas as pd


st.title('🎈 Iris Machine Learning App')

st.write('This is builds a machine learning model!')
with st.expander('Data'):
  st.write('**Raw Data**')
  csv_url = ('https://raw.githubusercontent.com/Joemusa/joe_machinelearning/refs/heads/master/Iris.csv')
  df = pd.read_csv(csv_url)
  df1 = df.drop(columns = ['Id'])
 
  
  features = df1.columns[:-1]
  
  X = df1[features]
  X
  y = df1['Species']
  y
  
with st.expander('Statistics'):
  st.write('description of the data')
  df1.describe()







                       





