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

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = [iris.target_names[i] for i in iris.target]

# Plot with plotly inside Streamlit
with st.expander("Data visualization"):
    fig = px.scatter(df,
                     x='sepal length (cm)',
                     y='sepal width (cm)',
                     color='target_name',
                     title='Sepal Dimensions by Species')
    st.plotly_chart(fig)
