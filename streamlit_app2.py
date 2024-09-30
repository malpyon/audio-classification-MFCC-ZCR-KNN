import streamlit as st
import pandas as pd
import numpy as np
import pickle
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from st_audiorec import st_audiorec
import librosa
import wave
import os

st.write("""
# KLASIFIKASI EMOSI WICARA MENGGUNAKAN METODE MFCC-ZCR-KNN
""")

datasetku = pd.read_csv('implementasi/audio_klas+me.csv')
x = datasetku.iloc[:, 2:-1].values
y = datasetku.iloc[:, -1].values

scaler = pickle.load(open('implementasi/scaler.pkl', 'rb'))
x_scaled = scaler.transform(x)
df_x_scaled = pd.DataFrame(data=x_scaled, columns=['MFCC'+str(x) for x in range(1,21)]+['ZCR'])

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.20, random_state = 0)
model = pickle.load(open('implementasi/model.pkl', 'rb'))
y_pred = model.predict(x_test)
cv_scores = cross_val_score(model, x_train, y_train, cv=10)
accuracy_score = pd.DataFrame(data=[cv_scores.mean()], columns=['accuracy'])

def get_feature(y,sr):
  feat_MFCC = [np.mean(val) for val in librosa.feature.mfcc(y=y, sr=sr)]
  feat_ZCR = [np.mean(librosa.feature.zero_crossing_rate(y=y))]
  features = feat_MFCC+feat_ZCR
  return features

def bytes_to_wav(byte_data, filename):
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(44100)
        wav_file.writeframes(byte_data)

st.write('record audio')
wav_audio_data = st_audiorec()

st.write("Atau")

audio_upload = st.file_uploader('pilih audio', type='.wav')

audio_file = None
if wav_audio_data is not None:
  audio_bytes = wav_audio_data
  bytes_to_wav(audio_bytes, 'output.wav')
  audio_file = 'output.wav'
elif audio_upload is not None:
  audio_file = audio_upload
  st.audio(audio_file)


if audio_file is not None:
    y, sr = librosa.load(audio_file, sr=None)
    feature = get_feature(y,sr)
    test_onefile = np.array([feature])
    scaled_onefile = scaler.transform(test_onefile)
    st.subheader('prediction')
    y_pred = model.predict(scaled_onefile)
    st.write(y_pred)
    file_path = 'output.wav'
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"The file {file_path} has been deleted.")


    


    