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

with st.sidebar:
  Sepal_length_cm = st.slider('Sepal length (cm)', 5.1, 7.9, 1.1)
  Sepal_width_cm = st.slider('Sepal width (cm)', 2.0, 4.4, 1.1)
  Petal_length_cm = st.slider('Sepal length (cm)', 1.0, 6.9, 1.1)
  Petal_width_cm = st.slider('Sepal width (cm)', 0.1, 2.4, 0.1)


                       





