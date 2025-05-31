import streamlit as st
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "lat": np.random.randn(1000) / 50 + 37.76,
    "lon": np.random.randn(1000) / 50 - 122.4,
    "size": np.random.randn(1000) * 100,
    "color": np.random.rand(1000, 4).tolist(),
})

st.title('스트림릿 지도')
st.map(df)
