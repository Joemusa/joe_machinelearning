# import plotly.express as px
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
iris_dataset = load_iris()
st.title('🎈 Iris Machine Learning App')

st.write('This is builds a machine learning model!')
with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.DataFrame(data=iris_dataset.data, columns=iris_dataset.feature_names)
  df['target'] = iris_dataset.target
  df
  
# Extract features (X) and target (y)
  X = df.drop(columns='target')  # Features (all columns except 'target')
  X
  y = df['target']               # Target (labels)
  y

st.title('Scatter Charts')
with st.expander('Petal length (cm) vs Petal width (cm)'):
  st.scatter_chart(data = df, x = 'petal length (cm)', y = 'petal width (cm)', color = 'target' )

with st.expander('Sepal length (cm) vs Sepal width (cm)'):
  st.scatter_chart(data = df, x = 'sepal length (cm)', y = 'sepal width (cm)', color = 'target' )






                       





