import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
iris_dataset = load_iris()
st.title('🎈 Machine Learning App')

st.write('This is app builds a machine learning model!')
iris_dataset['data'].shape
