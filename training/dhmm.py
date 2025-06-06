import numpy as np
import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dhmm import ECGBedDataset, ECGVectorQuantizer, ECGHMMSingleClassVQDetector, ECGHMMMultiClassVQDetector
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix


train_datas = []
for i in range(5):
    train_datas.append(ECGBedDataset(f'../dataset/train_class_{i}.csv'))

train_beats = []
for i in range(5):
    train_beats.append(np.stack([train_datas[i][j] for j in range(len(train_datas[i]))], axis=0))


detector_0 = ECGHMMSingleClassVQDetector(target_length=train_beats[0].shape[1], 
                                        n_states=8,
                                        window_size=20,
                                        hop_size=10,
                                        codebook_size=64,
                                        hmm_iter=100,
                                        normalize=False)

detector_1 = ECGHMMSingleClassVQDetector(target_length=train_beats[1].shape[1], 
                                        n_states=8,
                                        window_size=20,
                                        hop_size=10,
                                        codebook_size=64,
                                        hmm_iter=100,
                                        normalize=False)

detector_2 = ECGHMMSingleClassVQDetector(target_length=train_beats[2].shape[1],
                                        n_states=8,
                                        window_size=20,
                                        hop_size=10,
                                        codebook_size=64,
                                        hmm_iter=100,
                                        normalize=False)

detector_3 = ECGHMMSingleClassVQDetector(target_length=train_beats[3].shape[1],
                                        n_states=8,
                                        window_size=20,
                                        hop_size=10,
                                        codebook_size=64,
                                        hmm_iter=100,
                                        normalize=False)

detector_4 = ECGHMMSingleClassVQDetector(target_length=train_beats[4].shape[1],
                                        n_states=8,
                                        window_size=20,
                                        hop_size=10,
                                        codebook_size=64,
                                        hmm_iter=100,
                                        normalize=False)

models = [detector_0, detector_1, detector_2, detector_3, detector_4]



final_detector = ECGHMMMultiClassVQDetector(n_class=5, models=models)
final_detector.fit(train_beats)

with open('../checkpoints/dhmm.pkl', 'wb') as f:
    pickle.dump(final_detector, f)
