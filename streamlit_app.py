import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
iris_dataset = load_iris()
df = pd.DataFrame(data=bunch.data, columns=bunch.feature_names)
st.title('🎈 Machine Learning App')

st.write('This is app builds a machine learning model!')
df['target'] = bunch.target

print(df.head())
iris_dataset['data'].shape
