import plotly.express as px
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
iris_dataset = load_iris()
st.title('🎈 Machine Learning App')

st.write('This is app builds a machine learning model!')
with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.DataFrame(data=iris_dataset.data, columns=iris_dataset.feature_names)
  df['target'] = iris_dataset.target
  df
  st.write('**X**')
  y = df.target_names
  y

with st.sidebar:
  st.header('Input features')
  sepal = st.selectbox('sepal length', 'sepal width')
  petal = st.selectbox('petal length', 'petal width')
