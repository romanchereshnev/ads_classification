from helpers import get_Xy
import pickle
from typing import List
from fasttext_classifier import FastTextClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def create_train_test(data_dir="ml_interview_ads_data"):
    processed_data_list, binary_labels_lists = get_Xy(folder_name=data_dir, return_raw=False)
    train_indexes, test_indexes = train_test_split(np.arange(len(processed_data_list)))
    X_raw_train = [processed_data_list[i] for i in train_indexes]
    X_raw_test = [processed_data_list[i] for i in test_indexes]

    y_train_list = [binary_labels_lists[i] for i in train_indexes]
    y_test_list = [binary_labels_lists[i] for i in test_indexes]
    
    return X_raw_train, X_raw_test, y_train_list, y_test_list


def create_sentences_embeddings(X_list: List[List[str]], ft: FastTextClassifier) -> List[np.ndarray]:
    X_emb_sentences = []
    for sentenses in X_list:
        X_emb_sentences.append(ft.transform(sentenses))
    return X_emb_sentences


def create_ft_sent_vectors(X_raw_train, X_raw_test, y_train_list, y_test_list, filename,
                          epoch=200, minn=1, maxn=6, dim=100, model='skipgram'):
    flat_train_data = [x for X in X_raw_train for x in X]
    ft = FastTextClassifier(verbose=1, epoch=epoch, minn=minn, maxn=maxn, dim=dim, model=model)
    ft.fit(flat_train_data)
    
    X_train_list = create_sentences_embeddings(X_raw_train, ft)
    X_test_list = create_sentences_embeddings(X_raw_test, ft)

    with open(filename, 'wb') as f:
        pickle.dump( (X_train_list, X_test_list, y_train_list, y_test_list) , f)
        
        
def create_rec_data_from_long_matrix(x, y, time_steps, features):
    x_recs = []
    y_recs = []
    masks = []
    i = 0
    while i * time_steps < len(x):
        x_rec = np.zeros(dtype=np.float32, shape=(time_steps, features))
        y_rec = np.zeros(dtype=np.float32, shape=(time_steps,))
        mask = np.full(fill_value=False, dtype=np.bool, shape=(time_steps,))

        length = min([time_steps, len(x) - i * time_steps])
        x_rec[:length, :] = x[i*time_steps:i*time_steps + length, :]
        y_rec[:length] = y[i*time_steps:i*time_steps + length]
        mask[:length] = True
        i += 1
        x_recs.append(x_rec)
        y_recs.append(y_rec)
        masks.append(mask)
    
    return x_recs, y_recs, masks


def create_rec_data(X_list, y_list, time_steps):
    X_rec = []
    Y_rec = []
    mask_rec = []
    
    features = X_list[0].shape[1]
    
    for x, y in zip(X_list, y_list):
        x_recs, y_recs, masks = create_rec_data_from_long_matrix(x, y, time_steps, features)
        X_rec.extend(x_recs)
        Y_rec.extend(y_recs)
        mask_rec.extend(masks)
    X_rec = np.stack(X_rec)         
    Y_rec = np.stack(Y_rec)
    mask_rec = np.stack(mask_rec)
    return X_rec, Y_rec, mask_rec


def plot_loss_precision_recall(loss, val_loss, precision, val_precision, recall, val_recall):
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))

    ax[0, 0].plot(loss, label='train')
    ax[0, 0].plot(val_loss, label='val')
    ax[0, 0].set_title("loss")

    ax[0, 1].set_title("precision")
    ax[0, 1].plot(precision)
    ax[0, 1].plot(val_precision)

    ax[1, 1].set_title("recall")
    ax[1, 1].plot(recall)
    ax[1, 1].plot(val_recall)


    fig.legend()
    plt.show()
    
    
def make_flat(y_times, mask):
    y_flat = []

    for ys, ms in zip(y_times, mask):
        for y, m in zip(ys, ms):
            if m:
                y_flat.append(y)
    return np.array(y_flat)