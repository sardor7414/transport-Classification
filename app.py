import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
import platform

plt = platform.system()
if plt == 'Linux': 
    pathlib.WindowsPath = pathlib.PosixPath

st.title('Transportni klassifikatsiya qiluvchi model')

# Rasm yuklash tugmasi
file = st.file_uploader("Rasmni yuklang", type=['png', 'jpeg', 'jpg', 'svg'])

if file:
    st.image(file)
    img = PILImage.create(file)
    model = load_learner('trasnsport-model.pkl')
    model.predict(img)
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f'Ehtimollik: {probs[pred_id]*100:.1f}')
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
