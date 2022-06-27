import os
import sklearn
import numpy as np
import pandas as pd
from glob import glob
import scipy.io as sio
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns

MAX_LEN = 18286
MIN_LEN = 2714
LR = 1e-4
BATCH_SIZE = 30


def get_mat(mat_path):
    s = sio.loadmat(mat_path)
    return s['val'][0]


def preprocess(meta, to_path='precessed_data/normalized/', func_list=[], store=False):
    '''
    Perfrom the preprocessing to the data from given path and store the precessed data in given path

    Args:
        meta: like meta_pd which we created before
        from_path:  path contains *.mat data
        to_path: path where store the precessed data in npy form
        func_list : aprroaches , which will be performed on the data.Notice that,  the approaches will be performed in given ordering
        store: bool, True to store the data, False for only return tmp_data
    return:
    '''
    # tmp_meta = pd.DataFrame()
    # tmp_meta['id'] = meta['id']

    tmp_meta = meta.copy()
    # tmp_meta['to_path'] = to_path + meta['id'] + '.npy'

    tmp_meta['preprocessed_data'] = meta['data']
    for f in func_list:
        tmp_meta['preprocessed_data'] = tmp_meta['preprocessed_data'].map(f)

    if store == True:
        for path, array in zip(tmp_meta['to_path'].to_list(), tmp_meta['preprocessed_data'].to_list()[0]):
            np.save(path, array)

    # print(tmp_meta.head())
    return tmp_meta


def ecg_norm(ecg_np):
    '''
    Be used in preprocess() to perform normalization

    Args:
        ecg_np: ecg data in np.array form in shape of (N,)

    return:
        re: processed data in array form

    '''
    ecg_np = ecg_np.reshape(1, -1)
    re = preprocessing.Normalizer().fit_transform(ecg_np)[0]-0.5

    return re


def ecg_pad(ecg_np):
    '''
    Be used in preprocess() to perform padding (with mean by default )

    Args:
        ecg_np: ecg data in np.array form in shape of (N,)

    return:
        re: processed data in array form

    '''

    re = np.concatenate((ecg_np, np.ones(MAX_LEN - ecg_np.shape[0]) * ecg_np.mean()))

    return re


def ecg_pad_repeat(ecg_np):
    '''
    Be used in preprocess() to perform padding (with mean by default )

    Args:
        ecg_np: ecg data in np.array form in shape of (N,)

    return:
        re: processed data in array form

    '''

    re = np.pad(ecg_np, (0, MAX_LEN - ecg_np.shape[0]), 'wrap')
    # low the avg scroes

    return re


def ecg_fourier(ecg_np):
    '''
    Be used in preprocess() to perform Fourier Transform

    Args:
        ecg_np: ecg data in np.array form in shape of (N,)

    return:
        re: processed data in array form

    '''

    re = np.abs(np.fft.rfft(ecg_np))

    return re


def ecg_stand(ecg_np):
    '''
    Be used in preprocess() to perform Standardize

    Args:
        ecg_np: ecg data in np.array form in shape of (N,)

    return:
        re: processed data in array form

    '''

    re = (ecg_np - ecg_np.mean()) / ecg_np.std()
    return re


def cut_recording(ecg: np.array, length_threshold=9000):
    len_overlap = int(0.5 * length_threshold)
    start = 0
    end = 9000
    re = []
    re.append(ecg[start:end])
    while True:
        if end + len_overlap > len(ecg):
            re.append(ecg[-9000:])
            break
        re.append(ecg[start + len_overlap:end + len_overlap])
        start += len_overlap
        end += len_overlap

    # print(ecg)

    return re
def data_len_norm(meta):
    '''
    Normalize the lengh of data follwing the idea from paper:
    https://res.mdpi.com/d_attachment/sensors/sensors-20-02136/article_deploy/sensors-20-02136.pdf

    Args:

        meta: pd.Dataframe, meta date in pd form.

    return:
        re: meta data after  processing

    '''

    re_pd = meta.copy()
    # print(len(re_pd))
    for row in re_pd.iterrows():
        if row[1]['length'][0] > 9000:
            adding_list = cut_recording(row[1]['data'])

            for n in adding_list:
                new_recording = meta.iloc[row[0]].copy()
                new_recording['data'] = n
                #new_recording['id'] = 'extra_' + new_recording['id']
                new_recording['length'] = 9000
                # print(new_recording)
                re_pd = re_pd.append(new_recording, ignore_index=True)

            re_pd.drop(row[0], inplace=True)

        if row[1]['length'][0] < 9000:
            pass

    return re_pd


def ecg_len_norm(ecg_np):
    '''
    Be used in preprocess() to make the length be 9000 (over 9000 then cut, under 9000 then repeat)

    Args:
        ecg_np: ecg data in np.array form in shape of (N,)

    return:
        re: processed data in array form

    '''
    if len(ecg_np) < 9000:
        re = np.pad(ecg_np, (0, 9000 - ecg_np.shape[0]), 'reflect')
    elif len(ecg_np) > 9000:
        re = ecg_np[:9000]
    else:
        re = ecg_np

    return re
def data_ros(meta, cc_dict):
    '''
    Be used in preprocess() to perform Rondom over sampling. To address the imbalance problem.

    Args:
        cc_dict: dict, for cate_counts_dic
        meta: pd.Dataframe, meta date in pd form.

    return:
        re: meta data after ros

    '''

    data_aug_ratio_dic = dict(
        zip(cc_dict.keys(), (np.array([*cc_dict.values()]).max() / np.array([*cc_dict.values()])).round()))

    # print(data_aug_ratio_dic)
    re_pd = meta.copy()

    for (l, r) in data_aug_ratio_dic.items():
        # print(l,r)
        re_pd = re_pd.append([meta.loc[meta['label'] == l, :]] * int(r - 1), ignore_index=True)

    # print((re_pd['encoded_label'].value_counts().values.max()/re_pd['encoded_label'].value_counts().values).round())

    return re_pd


def pd_2array(X_pd, y_pd):
    '''
    Convert DataFrema to np.array


    Args:
        X_pd:X in datafrema form, each row is one np.array data
        y_pd: y in datafrema form, each row is a single number

    return:
        X,y in np.array form
    '''
    X_ = []
    y_ = []
    for X, y in zip(X_pd, y_pd):
        X_.append(X.tolist())
        y_.append(y)

    return np.array(X_), np.array(y_)


def RL(X_train_rl, y_train_rl, X_val_rl, y_val_rl, X_test_rl, y_test_rl):
    '''
    A Random forest classifier:

    '''
    clf = RandomForestClassifier(max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train_rl, y_train_rl)
    y_train_pred = clf.predict(X_train_rl)
    print("--------Random forest classifier--------")
    print("*****f1-score metrix on training set*****")
    print(classification_report(y_train_rl, y_train_pred))

    y_val_pred = clf.predict(X_val_rl)
    print("*****f1-score metrix on val set*****")
    print(classification_report(y_val_rl, y_val_pred))

    y_test_pred = clf.predict(X_test_rl)
    print("*****f1-score metrix on test set*****")
    print(classification_report(y_test_rl, y_test_pred))

    return clf


def MLP(X_train_mlp, y_train_mlp, X_val_mlp, y_val_mlp, X_test_mlp, y_test_mlp):
    '''
    A MLP classifier:

    '''
    clf = AdaBoostClassifier(n_estimators=100).fit(X_train_mlp, y_train_mlp)

    y_train_pred = clf.predict(X_train_mlp)
    print("--------MLP classifier--------")
    print("*****f1-score metrix on training set*****")
    print(classification_report(y_train_mlp, y_train_pred))

    y_val_pred = clf.predict(X_val_mlp)
    print("*****f1-score metrix on val set*****")
    print(classification_report(y_val_mlp, y_val_pred))

    y_test_pred = clf.predict(X_test_mlp)
    print("*****f1-score metrix on test set*****")
    print(classification_report(y_test_mlp, y_test_pred))

    return clf


class ecg_Dataset(Dataset):
    def __init__(self, X_pd, y_pd):
        """
        Args:

        """
        self.X_pd = X_pd
        self.y_pd = y_pd

    def __len__(self):
        return len(self.X_pd)

    def __getitem__(self, idx):
        return self.X_pd.iloc[idx], self.y_pd.iloc[idx]


# class MyModel(pl.LightningModule):
#     def __init__(
#             self,
#             num_inputs,
#             num_hidden_1,
#             num_hidden_2,
#             num_hidden_3,
#             num_hidden_4,
#             num_outputs,
#             dataloaders,
#     ):
#         super().__init__()
#
#         self.linear1 = nn.Linear(num_inputs, num_hidden_1)
#         self.ac1 = nn.ReLU()
#
#         self.linear2 = nn.Linear(num_hidden_1, num_hidden_2)
#         self.ac2 = nn.ReLU()
#
#         self.linear3 = nn.Linear(num_hidden_2, num_hidden_3)
#         self.ac3 = nn.ReLU()
#
#         self.linear4 = nn.Linear(num_hidden_3, num_hidden_4)
#         self.ac4 = nn.ReLU()
#
#         self.out = nn.Linear(num_hidden_4, num_outputs)
#
#         self.softmax = nn.Softmax(dim=1)
#
#         self.criterion = nn.CrossEntropyLoss()
#
#         self.train_accuracy = torchmetrics.F1Score()
#
#         self.val_accuracy = torchmetrics.F1Score()
#         self.test_accuracy = torchmetrics.F1Score()
#         self.dataloaders = dataloaders
#
#     def forward(self, inputs, labels=None):
#
#         inputs = inputs.float()
#
#         outputs = self.linear1(inputs)
#         outputs = self.ac1(outputs)
#
#         outputs = self.linear2(outputs)
#         outputs = self.ac2(outputs)
#
#         outputs = self.linear3(outputs)
#         outputs = self.ac3(outputs)
#
#         outputs = self.linear4(outputs)
#         outputs = self.ac4(outputs)
#
#         outputs = self.out(outputs)
#
#         outputs = self.softmax(outputs)
#
#         return outputs
#     def predict_step(self, batch, batch_idx):
#         inputs, labels = batch
#
#         labels = labels.long()
#
#         outputs = self(inputs)
#
#         # outputs = torch.argmax(outputs, dim=1)
#
#         preds = torch.argmax(outputs, dim=1)
#
#
#
#         return preds
#
#
#     def train_dataloader(self):
#         # ecg_train_dataset = ecg_Dataset(X_train_pd, y_train_pd)
#         # return DataLoader(ecg_train_dataset, batch_size=100)
#
#         return self.dataloaders[0]
#
#     def val_dataloader(self):
#         # ecg_val_dataset = ecg_Dataset(X_val_pd, y_val_pd)
#         # return DataLoader(ecg_val_dataset, batch_size=100)
#
#         return self.dataloaders[1]
#
#     def test_dataloader(self):
#         # ecg_test_dataset = ecg_Dataset(X_test_pd, y_test_pd)
#         # return DataLoader(ecg_test_dataset, batch_size=100)
#         return self.dataloaders[2]
#
#     def training_step(self, batch, batch_idx):
#         inputs, labels = batch
#
#         labels = labels.long()
#
#         outputs = self(inputs)
#
#         # outputs = torch.argmax(outputs, dim=1)
#
#         preds = torch.argmax(outputs, dim=1)
#         # print(preds)
#         self.train_accuracy(preds, labels)
#
#         # print("training acc:",self.train_accuracy(outputs, labels))
#         loss = self.criterion(outputs, labels)
#         # print("trainingh Loss:",loss)
#
#         self.log("train_loss", loss)
#
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         inputs, labels = batch
#         # inputs = inputs.long().view(inputs.size(0), -1)
#         labels = labels.long()
#
#         outputs = self(inputs)
#         # outputs = torch.argmax(outputs, dim=1)
#         preds = torch.argmax(outputs, dim=1)
#         self.val_accuracy(preds, labels)
#         # print("val acc:",self.val_accuracy(outputs, labels))
#         # print("total val acc:",self.val_accuracy.compute())
#
#         loss = self.criterion(outputs, labels)
#         # print("VAL Loss:",loss)
#
#         self.log("val_loss", loss)
#
#     def validation_epoch_end(self, outs):
#         print("total val acc:", self.val_accuracy.compute())
#         print("*" * 25)
#
#     def training_epoch_end(self, outs):
#         # print(   compute_epoch_loss_from_outputs(outs))
#         print("total training acc:", self.train_accuracy.compute())
#
#     def configure_optimizers(self):
#         decayRate = 0.96
#         optimizer = Adam(self.parameters(), lr=LR)
#         scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
#
#         return [optimizer], [scheduler]
#
#     def test_step(self, batch, batch_idx):
#         inputs, labels = batch
#         # inputs = inputs.long().view(inputs.size(0), -1)
#         labels = labels.long()
#
#         outputs = self(inputs)
#         # outputs = torch.argmax(outputs, dim=1)
#         preds = torch.argmax(outputs, dim=1)
#
#         # print(preds)
#
#         self.test_accuracy(preds, labels)
#
#     def test_epoch_end(self, outs):
#         # print(   compute_epoch_loss_from_outputs(outs))
#         print("total test acc:", self.test_accuracy.compute())

class MyModel(pl.LightningModule):
    def __init__(
            self,
            cin,
            cout,

            dataloaders,
    ):
        super().__init__()

        self.cin = cin,
        self.cout = cout
        self.seq_fft = nn.Sequential(
            nn.Conv1d(1, 32, 5),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 32, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 64, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 128, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 256, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.5),

            nn.Conv1d(256, 512, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(512, 512, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.5),

            nn.Flatten(),
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.cout),
            nn.Softmax(dim=1)

        )

        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracy = torchmetrics.F1Score()

        self.val_accuracy = torchmetrics.F1Score()
        self.test_accuracy = torchmetrics.F1Score()
        self.dataloaders = dataloaders

    def forward(self, inputs, labels=None):
        inputs = inputs.float()

        outputs = self.seq_fft(inputs)

        return outputs

    def predict_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1])
        # inputs = inputs.long().view(inputs.size(0), -1)
        labels = labels.long()


        outputs = self(inputs)

        # outputs = torch.argmax(outputs, dim=1)

        preds = torch.argmax(outputs, dim=1)

        return preds

    def train_dataloader(self):
        # ecg_train_dataset = ecg_Dataset(X_train_pd, y_train_pd)
        # return DataLoader(ecg_train_dataset, batch_size=100)

        return self.dataloaders[0]

    def val_dataloader(self):
        # ecg_val_dataset = ecg_Dataset(X_val_pd, y_val_pd)
        # return DataLoader(ecg_val_dataset, batch_size=100)

        return self.dataloaders[1]

    def test_dataloader(self):
        # ecg_test_dataset = ecg_Dataset(X_test_pd, y_test_pd)
        # return DataLoader(ecg_test_dataset, batch_size=100)
        return self.dataloaders[2]

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1])
        # print(inputs.shape)
        labels = labels.long()

        outputs = self(inputs)

        # outputs = torch.argmax(outputs, dim=1)

        preds = torch.argmax(outputs, dim=1)
        # print(preds)
        self.train_accuracy(preds, labels)

        # print("training acc:",self.train_accuracy(outputs, labels))
        loss = self.criterion(outputs, labels)
        # print("trainingh Loss:",loss)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1])
        # inputs = inputs.long().view(inputs.size(0), -1)
        labels = labels.long()

        outputs = self(inputs)
        # outputs = torch.argmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        self.val_accuracy(preds, labels)
        # print("val acc:",self.val_accuracy(outputs, labels))
        # print("total val acc:",self.val_accuracy.compute())

        loss = self.criterion(outputs, labels)
        # print("VAL Loss:",loss)

        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1])

        # inputs = inputs.long().view(inputs.size(0), -1)
        labels = labels.long()

        outputs = self(inputs)
        # outputs = torch.argmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

        # print(preds)

        self.test_accuracy(preds, labels)

    def validation_epoch_end(self, outs):
        print("total val acc:", self.val_accuracy.compute())
        print("*" * 25)

    def training_epoch_end(self, outs):
        # print(   compute_epoch_loss_from_outputs(outs))
        print("total training acc:", self.train_accuracy.compute())

    def test_epoch_end(self, outs):
        # print(   compute_epoch_loss_from_outputs(outs))
        print("total test acc:", self.test_accuracy.compute())

    def configure_optimizers(self):
        decayRate = 0.96
        optimizer = Adam(self.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

        return [optimizer], [scheduler]


def main():
    ## Load data and create meta data
    training_data_dir = "../training/"
    training_data_path = glob(training_data_dir + "*mat")
    training_path_id_dic = {x.split('/')[-1].split('.')[0]: x for x in training_data_path}

    reference = pd.read_csv(training_data_dir + "REFERENCE.csv", header=None)
    reference = reference.rename(columns={0: 'id', 1: "label"})
    reference_dic = dict(zip(reference['id'].to_list(), reference['label'].to_list()))

    meta_pd = pd.DataFrame(columns=["id", "path", "label"])
    meta_pd['id'] = training_path_id_dic.keys()
    meta_pd['path'] = meta_pd['id'].map(training_path_id_dic.get)
    meta_pd['label'] = meta_pd['id'].map(reference_dic.get)
    meta_pd['encoded_label'] = pd.Categorical(meta_pd['label']).codes
    meta_pd['data'] = meta_pd['path'].map(get_mat)
    meta_pd['mean'] = meta_pd['data'].map(np.mean)
    meta_pd['std'] = meta_pd['data'].map(np.std)
    meta_pd['length'] = meta_pd['data'].map(np.shape)

    # print(meta_pd.head(10))

    ## Do the length Normalization
    tmp_pd = data_len_norm(meta_pd)
    meta_pd = tmp_pd

    ## Data Visualization: ratio of labels
    cates = pd.Categorical(meta_pd['label'], ordered=True).categories
    cate_counts = meta_pd['encoded_label'].value_counts().to_list()
    cate_counts_dic = dict(zip(cates, [len(meta_pd[meta_pd['label'] == i]) for i in cates]))
    plt.pie(cate_counts_dic.values(), labels=cate_counts_dic.keys(), colors=sns.color_palette('pastel')[0:4],
            autopct='%.0f%%')
    plt.show()
    fig, axs = plt.subplots()
    axs.bar(cate_counts_dic.keys(), cate_counts_dic.values())

    ##length of data
    # MAX_LEN = meta_pd["length"].max()[0]  # 18286
    # MIN_LEN = meta_pd["length"].min()[0]  ## 2714

    ## Data preprocessing
    meta_precessed_pd = preprocess(meta=meta_pd, func_list=[ecg_len_norm, ecg_norm])

    ### Split the data into training , val and test set
    train_pd, test_pd = train_test_split(meta_precessed_pd, test_size=0.3, random_state=42,
                                         stratify=meta_precessed_pd['encoded_label'])

    train_pd_ros = data_ros(train_pd, cate_counts_dic)
    X_train_pd = train_pd_ros['preprocessed_data']
    y_train_pd = train_pd_ros['encoded_label']
    X_val_pd, X_test_pd, y_val_pd, y_test_pd = train_test_split(test_pd['preprocessed_data'], test_pd['encoded_label'],
                                                                test_size=0.33, random_state=42,
                                                                stratify=test_pd['encoded_label'])
    X_train, y_train = pd_2array(X_train_pd, y_train_pd)
    X_val, y_val = pd_2array(X_val_pd, y_val_pd)
    X_test, y_test = pd_2array(X_test_pd, y_test_pd)

    ### Test on some simple ML models
    ##
    # rl_clf = RL(X_train, y_train, X_val, y_val, X_test, y_test)
    # mlp_clf= MLP(X_train, y_train, X_val, y_val, X_test, y_test)

    ### Test on  NN using torch

    ecg_train_dataset = ecg_Dataset(X_train_pd, y_train_pd)
    ecg_train_dataloader = DataLoader(ecg_train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    ecg_val_dataset = ecg_Dataset(X_val_pd, y_val_pd)
    ecg_val_dataloader = DataLoader(ecg_val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    ecg_test_dataset = ecg_Dataset(X_test_pd, y_test_pd)
    ecg_test_dataloader = DataLoader(ecg_test_dataset, batch_size=BATCH_SIZE, shuffle=True)



    model = MyModel(1, 4, [ecg_train_dataloader, ecg_val_dataloader, ecg_test_dataloader])
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=100, gpus=0)  #

    # trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=1000, gpus=0,callbacks=[EarlyStopping(monitor="val_loss", mode="min")]) #
    trainer.fit(model)
    torch.save(model.state_dict(),"mymodel")
    trainer.validate(model)
    trainer.test(model)

    # model = torch.load("tmp/saved_model")
    # trainer = pl.Trainer(accelerator="gpu", devices=1, gpus=0)
    # trainer.test(model)


if __name__ == '__main__':
    main()