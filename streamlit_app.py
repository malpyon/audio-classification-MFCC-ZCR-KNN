import streamlit as st
import pandas as pd
import numpy as np
import pickle
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


st.write("""
# MFCC & ZCR audio classification with K-NN
""")

# st.subheader('feature extraction dataset')

datasetku = pd.read_csv('audio_klas+me.csv')
# st.write(datasetku)

# st.subheader('scaled dataset')

x = datasetku.iloc[:, 2:-1].values
y = datasetku.iloc[:, -1].values

scaler = pickle.load(open('scaler.pkl', 'rb'))
x_scaled = scaler.transform(x)
df_x_scaled = pd.DataFrame(data=x_scaled, columns=['MFCC'+str(x) for x in range(1,21)]+['ZCR'])
# st.write(df_x_scaled)

# st.subheader('accuracy with k-nn')

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.20, random_state = 0)
model = pickle.load(open('model.pkl', 'rb'))
y_pred = model.predict(x_test)
cv_scores = cross_val_score(model, x_train, y_train, cv=10)
accuracy_score = pd.DataFrame(data=[cv_scores.mean()], columns=['accuracy'])

# st.write(accuracy_score)

#emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
def get_feature(y,sr):
  feat_MFCC = [np.mean(val) for val in librosa.feature.mfcc(y=y, sr=sr)]
  feat_ZCR = [np.mean(librosa.feature.zero_crossing_rate(y=y))]
  features = feat_MFCC+feat_ZCR
  return features

audio_file = st.file_uploader('chose audio', type='.wav')
if audio_file is not None:
    # file_name = audio_file.name
    # audio_play = st.audio(audio_file)
    # searchfor = datasetku[datasetku['file_name'] == file_name]
    # st.subheader('file info')
    # st.write(searchfor)
    y, sr = librosa.load(audio_file, sr=None)
    feature = get_feature(y,sr)
    test_onefile = np.array([feature])
    scaled_onefile = scaler.transform(test_onefile)
    st.audio(audio_file)
    st.subheader('pediction')
    y_pred = model.predict(scaled_onefile)
    st.write(y_pred)




