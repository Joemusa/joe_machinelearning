import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
iris_dataset = load_iris()
df = pd.DataFrame(data=Bunch.data, columns=Bunch.feature_names)
st.title('🎈 Machine Learning App')

st.write('This is app builds a machine learning model!')
df['target'] = Bunch.target

print(df.head())
iris_dataset['data'].shape
