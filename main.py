from helpers import create_and_save_sent_data, create_y_list, make_flat

import numpy as np

import pickle

from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import confusion_matrix
from keras.models import Model, Input
from keras.layers import GRU, Dense, TimeDistributed, Bidirectional, Dropout
from keras.models import load_model
import tensorflow as tf
from os.path import exists



def create_rec_data(X_list, y_list, time_steps):
    """
    Create data for RNN from sequences.
    """
    X_rec = []
    Y_rec = []
    mask_rec = []

    features = X_list[0].shape[1]

    for x, y in zip(X_list, y_list):
        i = 0

        while i * time_steps < len(x):
            x_rec = np.zeros(dtype=np.float32, shape=(time_steps, features))
            y_rec = np.zeros(dtype=np.float32, shape=(time_steps,))
            masks = np.full(fill_value=False, dtype=np.bool, shape=(time_steps,))

            length = min([time_steps, len(x) - i * time_steps])
            # print(length)
            x_rec[:length, :] = x[:length, :]
            y_rec[:length] = y[:length]
            masks[:length] = True
            i += 1

            X_rec.append(x_rec)
            Y_rec.append(y_rec)
            mask_rec.append(masks)
    X_rec = np.stack(X_rec)
    Y_rec = np.stack(Y_rec)
    mask_rec = np.stack(mask_rec)
    return X_rec, Y_rec, mask_rec


def crate_model(timesteps, features):
    seed = 1
    np.random.seed(seed)
    tf.set_random_seed(seed)
    input = Input(shape=(timesteps, features))
    model = Dropout(0.1)(input)
    model = Bidirectional(GRU(units=200, return_sequences=True, recurrent_dropout=0.1))(model)
    model = TimeDistributed(Dense(100, activation="sigmoid"))(model)
    out = TimeDistributed(Dense(1, activation="sigmoid"))(model)
    model = Model(input, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


if __name__ == '__main__':

    # LOAD PRE-TRAINED MODEL
    load = True
    # load = False

    filename = 'sentences.pkl'

    if not exists(filename):
        print("Creating data")
        create_and_save_sent_data(filename)

    print("load data")
    with open(filename, 'rb') as f:
        X_sent_train_list, X_sent_test_list, y_train, y_test = pickle.load(f)

    y_train_list = create_y_list(X_sent_train_list, y_train)
    y_test_list = create_y_list(X_sent_test_list, y_test)

    timesteps = 50
    features = 100

    X_train_rec, y_train_rec, mask_train_rec = create_rec_data(X_sent_train_list, y_train_list, timesteps)
    X_test_rec, y_test_rec, mask_test_rec = create_rec_data(X_sent_test_list, y_test_list, timesteps)

    y_train_rec = y_train_rec[:, :, None]
    y_test_rec = y_test_rec[:, :, None]

    model = crate_model(timesteps, features)

    if load and exists("model"):
        print("loading model")
        model = load_model("model")

    else:
        print("fitting model")
        model.fit(X_train_rec, y_train_rec, batch_size=500, epochs=500, validation_data=[X_test_rec, y_test_rec])
    y_pred_probs = model.predict(X_test_rec)

    y_pred = np.round(y_pred_probs)
    y_flat_pred = make_flat(y_pred, mask_test_rec)
    y_flat_true = make_flat(y_test_rec, mask_test_rec)

    print("Results: \n")
    print("F1:", f1_score(y_flat_true, y_flat_pred))
    print(classification_report(y_flat_true, y_flat_pred))
    print(confusion_matrix(y_flat_true, y_flat_pred))

    model.save("model")




