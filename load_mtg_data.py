import pickle
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import madmom
import librosa
from scipy import signal
SAMPLE_RATE = 44100

"""Create Element to Int and Int to Element dictionaries from a list of elements"""


def create_dict(sequence):
    return (dict((element, number) for number, element in enumerate(sequence)),
            dict((number, element) for number, element in enumerate(sequence)))

def load_data(dictionary, path='giantsteps-mtg-key-dataset', augmentation=True):
    os.chdir(path)
    infos = pd.read_csv("annotations/annotations.txt", sep=None,engine='python')
    os.chdir('audio')
    audios = []
    keys = []
    counter = 0
    for id, key,confidence in zip(infos['ID'], infos['MANUAL KEY'],infos['C']):
        #if counter==10:
        #    break
        if confidence>1:
            counter += 1
            if '/' not in key and '-' not in key :
                if 'major' in key:
                    key = key[:key.find('major') + 5].strip()
                elif 'minor' in key:
                    key = key[:key.find('minor') + 5].strip()
                for f in os.listdir(os.curdir):
                    if str(id) in f:
                        audio = madmom.audio.Signal(f, num_channels=1, sample_rate=SAMPLE_RATE)
                        for transpose in [ -2, -1,0, 1, 2]:

                            new_key= dictionary[key]+transpose
                            if 'major' in key:
                                if new_key<0:
                                    new_key+=12
                                if new_key>11:
                                    new_key-=12
                            elif 'minor' in key:
                                if new_key<12:
                                    new_key+=12
                                if new_key>23:
                                    new_key-=12

                            shift = librosa.effects.pitch_shift(np.array(audio,dtype=float), sr=SAMPLE_RATE, n_steps=transpose)
                            length = int(shift.shape[0])
                            idx = 0
                            frame_length = 30  # in seconds
                            while (idx + frame_length) * SAMPLE_RATE <= length:
                                segment = shift[idx * SAMPLE_RATE:(idx + frame_length) * SAMPLE_RATE]
                                idx += frame_length
                                """stft = librosa.stft(segment, n_fft=8192)
                                cqt= np.abs(librosa.cqt(segment, sr=SAMPLE_RATE, fmin=librosa.note_to_hz('C2'),
                                                   n_bins=48, bins_per_octave=12))
                                chroma = librosa.feature.chroma_cqt(C=cqt,bins_per_octave=12,n_octaves=4)
                                chroma2 = np.array(librosa.feature.chroma_cqt(segment,sr=SAMPLE_RATE))"""

                                chroma = np.array(librosa.feature.chroma_cqt(y=segment,sr=SAMPLE_RATE,hop_length=8820))

                                keys.append(new_key)
                                audios.append(chroma)
    os.chdir("..")
    os.chdir("..")
    return (audios, keys)

def create_test_split(x,y,split=0.1):
    size= int(len(x)*split)
    x_test=[]
    y_test=[]
    for i in range(size):
        a= np.random.randint(low=0,high=len(x))
        x_test.append(x[a])
        y_test.append(y[a])
        x.pop(a)
        y.pop(a)

    return x,y,x_test,y_test


keys = ['C major', 'C# major', 'D major', 'D# major', 'E major', 'F major', 'F# major', 'G major', 'G# major',
        'A major', 'A# major', 'B major',
        'C minor', 'C# minor', 'D minor', 'D# minor', 'E minor', 'F minor', 'F# minor', 'G minor', 'G# minor',
        'A minor', 'A# minor', 'B minor']

maj_keys = ['C major', 'C# major', 'D major', 'D# major', 'E major', 'F major', 'F# major', 'G major', 'G# major',
        'A major', 'A# major', 'B major']

min_keys=['C minor', 'C# minor', 'D minor', 'D# minor', 'E minor', 'F minor', 'F# minor', 'G minor', 'G# minor',
        'A minor', 'A# minor', 'B minor']
key_int, int_key = create_dict(keys)
x, y = load_data(key_int)
y = tf.one_hot(y, len(keys))
with open('mgt30sec.pickle', 'wb') as f:
    pickle.dump([x,y], f)
