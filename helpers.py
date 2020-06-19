import numpy as np
import os.path
import string
from fasttext_classifier import FastTextClassifier
from sklearn.model_selection import train_test_split
import pickle

def load_data(folder_name:str = "ml_interview_ads_data"):
    data_lists = []
    labels_lists = []

    for file in sorted(os.listdir(folder_name)):
        with open(os.path.join(folder_name, file), 'rt', encoding='utf-8-sig') as f:
            lines = f.readlines()
        data = []
        labels = []

        for line in lines:

            label, _, text = line.partition(',')
            data.append(text)
            labels.append(label)

        data_lists.append(data)
        labels_lists.append(labels)
        
    return data_lists, labels_lists


def transform_labels(labels_lists):
    binary_labels_list = []
    for labels in labels_lists:
        binary_labels_list.append(np.array([1 if l != 'O' else 0 for l in labels]))
    return binary_labels_list


def process_data(raw_data_list):
    processed_data_list = []
    
    for raw_data in raw_data_list:
        processed_data_list.append([
            " ".join(l.lower().translate(str.maketrans('', '', string.punctuation)).split())
            for l in raw_data]
        )
    return processed_data_list


def get_Xy(folder_name:str = "ml_interview_ads_data", return_raw: bool = False):
    raw_data_lists, labels_lists = load_data(folder_name=folder_name)
    binary_labels_lists = transform_labels(labels_lists)
    processed_data_list = process_data(raw_data_lists)
    if return_raw:
        return processed_data_list, binary_labels_lists, return_raw 
    else:
        return processed_data_list, binary_labels_lists

    
def create_data_from_list(data_list, indexes, labels_list):
    X_raw = [] 
    y = []
    for i in indexes:
        X_raw.extend( data_list[i] )
        y.extend(labels_list[i] )

    y = np.array(y)
    
    return X_raw, y
  
    
def create_train_test_data(processed_data_list, binary_labels_lists, train_indexes, test_indexes):
    X_raw_train, y_train = create_data_from_list(processed_data_list, train_indexes, binary_labels_lists)
    X_raw_test, y_test = create_data_from_list(processed_data_list, test_indexes, binary_labels_lists)
    return X_raw_train, X_raw_test, y_train, y_test


def create_and_save_sent_data(filename="sentences2.pkl", folder_name="ml_interview_ads_data"):
    """Create sentences vectors based on FastText"""

    processed_data_list, binary_labels_lists = get_Xy(return_raw=False)
    train_indexes, test_indexes = train_test_split(np.arange(len(processed_data_list)))
    X_raw_train, X_raw_test, y_train, y_test = create_train_test_data(processed_data_list, binary_labels_lists, train_indexes, test_indexes)

    ft = FastTextClassifier(verbose=1, epoch=200, minn=1, maxn=6, model='skipgram')
    ft.fit(X_raw_train)

    X_sent_train = []

    for i in train_indexes:
        sentences = processed_data_list[i]
        print(len(sentences))
        X_sent_train.append(np.array(ft.transform(sentences)))

    X_sent_test = []

    for i in test_indexes:
        sentences = processed_data_list[i]
        print(len(sentences))
        X_sent_test.append(np.array(ft.transform(sentences)))

    with open(filename, 'wb') as f:
        pickle.dump((X_sent_train, X_sent_test, y_train, y_test), f)

    return X_sent_train, X_sent_test, y_train, y_test


def make_flat(y_times, mask):
    """Create flat array from 3d matrix based on mask"""
    y_flat = []

    for ys, ms in zip(y_times, mask):
        for y, m in zip(ys, ms):
            if m:
                y_flat.append(y)
    return np.array(y_flat)


def create_y_list(X_list, y):
    y_list = list()

    length = 0
    for x in X_list:
        y_list.append(y[length:length+len(x)])
        length += len(x)

    return y_list