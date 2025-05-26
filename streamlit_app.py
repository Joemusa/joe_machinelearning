# import plotly.express as px
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
with st.sidebar:
  st.header('Input Features')
  sepal = st.selectbox('Sepal',('sepal length','sepal width'))
  petal = st.selectbox('Petal',('petal length','petal width'))

with st.expander('Data Visualization'):
  st.scatter_chart(df)






