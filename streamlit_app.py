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
  sepal_length_(cm) = st.slider('Sepal length (cm)',5.1, 7.9, 1)

with st.expander('Data Visualization'):
  st.scatter_chart(data = df, x = 'petal length (cm)', y = 'petal width (cm)', color = 'target' )
  
  






