# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""

import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os
from typing import List, Tuple


import numpy as np
import pandas as pd
from glob import glob
import scipy.io as sio
from sklearn import preprocessing

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from train import *





###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : str='model.npy',is_binary_classifier : bool=False) -> List[Tuple[str,str]]:
    '''
    Parameters
    ----------
    model_name : str
        Dateiname des Models. In Code-Pfad
    ecg_leads : list of numpy-Arrays
        EKG-Signale.
    fs : float
        Sampling-Frequenz der Signale.
    ecg_names : list of str
        eindeutige Bezeichnung für jedes EKG-Signal.
    model_name : str
        Name des Models, kann verwendet werden um korrektes Model aus Ordner zu laden
    is_binary_classifier : bool
        Falls getrennte Modelle für F1 und Multi-Score trainiert werden, wird hier übergeben, 
        welches benutzt werden soll
    Returns
    -------
    predictions : list of tuples
        ecg_name und eure Diagnose
    '''

#------------------------------------------------------------------------------
# Euer Code ab hier
    ## Load data and create meta data

    # print(ecg_leads)
    # test_data_dir = "../test/"
    # test_data_path = glob(test_data_dir + "*mat")
    # test_path_id_dic = {x.split('/')[-1].split('.')[0]: x for x in test_data_path}

    # reference = pd.read_csv(test_data_dir + "REFERENCE.csv", header=None)
    # reference = reference.rename(columns={0: 'id', 1: "label"})
    # reference_dic = dict(zip(reference['id'].to_list(), reference['label'].to_list()))
    try:
        meta_pd = pd.DataFrame(columns=[ "id","data"])
        meta_pd['id'] = ecg_names
        meta_pd['data'] = ecg_leads
        # print(meta_pd.head())
        # meta_pd = pd.DataFrame(columns=["id", "path", "label"])
        # meta_pd['id'] = test_path_id_dic.keys()
        # meta_pd['path'] = meta_pd['id'].map(test_path_id_dic.get)
        # meta_pd['label'] = meta_pd['id'].map(reference_dic.get)
    #    meta_pd['encoded_label'] = pd.Categorical(meta_pd['label']).codes
        meta_pd['encoded_label'] = 0
        # meta_pd['data'] = meta_pd['path'].map(get_mat)
        meta_pd['mean'] = meta_pd['data'].map(np.mean)
        meta_pd['std'] = meta_pd['data'].map(np.std)
        meta_pd['length'] = meta_pd['data'].map(np.shape)
        # print(meta_pd.head().data)


        meta_precessed_pd = preprocess(meta=meta_pd, func_list=[ecg_len_norm, ecg_norm, ])

        print(meta_precessed_pd.head())

        ecg_test_dataset = ecg_Dataset(meta_precessed_pd['preprocessed_data'], meta_precessed_pd['encoded_label'])
        ecg_test_dataloader = DataLoader(ecg_test_dataset, batch_size=10)


        #saved_model = MyModel(4501, 3000, 2000, 1000, 500, 4, [])
        saved_model = MyModel(1, 4,[])
        saved_model.load_state_dict(torch.load(model_name))
        trainer = pl.Trainer(accelerator="gpu", devices=1, gpus=0)
        preds = trainer.predict(saved_model, dataloaders=ecg_test_dataloader)
        predictions = list()
        true_labels = ['A','N','O','~']
        # print(preds)
        for idx, pred in enumerate(*preds):
            # print(pred)
            predictions.append((meta_precessed_pd['id'].iloc[idx], true_labels[idx]))

        return predictions
        # print(predictions)
    
    except Exception as e:
            #print something
            print(e)


    # predictions.append((ecg_names[idx], 'A'))
#------------------------------------------------------------------------------    
    return predictions # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
                               
                               
        
