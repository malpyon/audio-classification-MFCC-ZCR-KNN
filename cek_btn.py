import streamlit as st
import numpy as np
from st_audiorec import st_audiorec

tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"])
data = np.random.randn(10, 1)

tab1.subheader("A tab with a chart")
tab1.wav_audio_data = st_audiorec()

tab2.subheader("A tab with the data")
tab2.audio_file = st.file_uploader('chose audio', type='.wav')

if tab1.audio_file is not None:
    st.write("olo")