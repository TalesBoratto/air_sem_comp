# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 11:29:56 2022

@author: tbora
"""

import numpy as np
import librosa
import librosa.display
from pathlib import Path
import matplotlib.pyplot as plt

#%% PRÁTICA 1
#Leitura via librosa
bumbo = [
    librosa.load(p, sr=44100)[0] for p in Path().glob('./bumbo/kick_*.mp3')
]
caixa = [
    librosa.load(p, sr=44100)[0] for p in Path().glob('./caixa/snare_*.mp3')
]

#Plotando áudios Bumbo
plt.figure(figsize=(15, 6))
for i, x in enumerate(bumbo):
    plt.subplot(2, 5, i+1)
    librosa.display.waveplot(x[:20000])
    plt.ylim(-1, 1)
    
#Plotando áudios Caixa
plt.figure(figsize=(15, 6))
for i, x in enumerate(caixa):
    plt.subplot(2, 5, i+1)
    librosa.display.waveplot(x[:20000])
    plt.ylim(-1, 1)
    
    
def extrair_atributos(sinal):
    return [
        #tsfel já calcular no sinal inteiro => 1 janela
        tsfel.feature_extraction.features.zero_cross(sinal),
        tsfel.feature_extraction.features.spectral_centroid(sinal, fs=44100),
        #Librosa também consegue calcular, mas utiliza janelamento no sinal
        #librosa.feature.zero_crossing_rate(signal)[0, 0],
        #librosa.feature.spectral_centroid(signal)[0, 0],
    ]

atributos_bumbo = np.array([extrair_atributos(x) for x in bumbo])
atributos_caixa = np.array([extrair_atributos(x) for x in caixa])

plt.figure(figsize=(14, 5))
plt.hist(atributos_bumbo[:,0], color='b', range=(0, 0.2), alpha=0.5, bins=20)
plt.hist(atributos_caixa[:,0], color='r', range=(0, 0.2), alpha=0.5, bins=20)
plt.legend(('bumbos', 'caixas'))
plt.xlabel('Zero Crossing Rate')
plt.ylabel('Count')

plt.figure(figsize=(14, 5))
plt.hist(atributos_bumbo[:,1], color='b', range=(0, 4000), bins=30, alpha=0.6)
plt.hist(atributos_caixa[:,1], color='r', range=(0, 4000), bins=30, alpha=0.6)
plt.legend(('kicks', 'snares'))
plt.xlabel('Spectral Centroid (frequency bin)')
plt.ylabel('Count')


#%% PRÁTICA 2

from sklearn.decomposition import PCA
import pandas as pd

def audio_read(num_audios=30, fs=44100, sec=21, c1=10, c3=10, c4=10):

    X = np.zeros(( fs*sec , num_audios))
    idx = 0
    
    #B8_novo
    for i in range(1,c1+1):
        print("B8 Novo - " + str(i))
        audio, fs = librosa.load('./b8_novo/'+str(i)+'.wav', sr=fs)

        if audio.shape[0] < 926100:
            dado = audio[:]
            dado = np.pad(dado, (0, 926100 - audio.shape[0]))
            X[:,idx] = dado/dado.max()
            
        elif audio.shape[0] > 926100:
            X[:,idx] = audio[0:926100]/audio[0:926100].max()
            
        else:
            X[:,idx] = audio[:]/audio[:].max()
    
        idx += 1
        
    #B10
    for i in range(1,c3+1):
        print("B10 - " + str(i))
        audio, fs = librosa.load('./b10/'+str(i)+'.wav', sr=fs)

        if audio.shape[0] < 926100:
            dado = audio[:]
            dado = np.pad(dado, (0, 926100 - audio.shape[0]))
            X[:,idx] = dado/dado.max()
            
        elif audio.shape[0] > 926100:
            X[:,idx] = audio[0:926100]/audio[0:926100].max()
            
        else:
            X[:,idx] = audio[:]/audio[:].max()
            
        idx += 1
        
    
    #B20
    for i in range(1,c4+1):
        print("B20 - " + str(i))
        audio, fs = librosa.load('./b20/'+str(i)+'.wav', sr=fs)

        if audio.shape[0] < 926100:
            dado = audio[:]
            dado = np.pad(dado, (0, 926100 - audio.shape[0]))
            X[:,idx] = dado/dado.max()
            
        elif audio.shape[0] > 926100:
            X[:,idx] = audio[0:926100]/audio[0:926100].max()
            
        else:
            X[:,idx] = audio[:]/audio[:].max()
            
        idx += 1
            
    return pd.DataFrame(X)


audios = audio_read()

num_audios = 30
n_coeff = 13
tsfel_mfcc = np.zeros (( num_audios, n_coeff))

for i in range(num_audios):
  tsfel_mfcc[i,:] = tsfel.feature_extraction.features.mfcc(X.iloc[:,i], fs=44100, pre_emphasis=1.00, nfft=926100, nfilt=40, num_ceps=13, cep_lifter=22)

pca = PCA(n_components=2)
pca_features = pca.fit_transform(tsfel_mfcc)

plt.scatter(pca_features[0:10,0], pca_features[0:10,1], color='blue')
plt.scatter(pca_features[10:20,0], pca_features[10:20,1], color='red')
plt.scatter(pca_features[20:30,0], pca_features[20:30,1], color='yellow')