import pandas as pd
import numpy as np
import plotly.graph_objects as go

from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split


def visualize_training(history):
    fig = go.Figure()
    fig.add_trace(go.Scattergl(y=history.history['loss'],
                               name='Train'))
    fig.add_trace(go.Scattergl(y=history.history['val_loss'],
                               name='Valid'))
    fig.update_layout(height=500, width=700,
                      xaxis_title='Epoch',
                      yaxis_title='Loss')
    fig.show()


class NN(object):
    def __init__(self, dataset_path, inputs):
        self.db = pd.read_excel(dataset_path)
        self.db_features = pd.DataFrame()
        self.db_outputs = pd.DataFrame()
        self.db_features['fo'] = pd.to_numeric(self.db['fo'], errors='coerce')
        self.db_features['bandwidth'] = pd.to_numeric(self.db['bandwidth'], errors='coerce')
        self.db_features['length'] = pd.to_numeric(self.db['length'], errors='coerce')
        self.db_features['mclin'] = pd.to_numeric(self.db['mclin'], errors='coerce')
        self.db_outputs['w1'] = pd.to_numeric(self.db['w1'], errors='coerce')
        self.db_outputs['w2'] = pd.to_numeric(self.db['w2'], errors='coerce')
        self.db_outputs['w3'] = pd.to_numeric(self.db['w3'], errors='coerce')
        self.db_outputs['s1'] = pd.to_numeric(self.db['s1'], errors='coerce')
        self.db_outputs['s2'] = pd.to_numeric(self.db['s2'], errors='coerce')
        self.db_outputs['s3'] = pd.to_numeric(self.db['s3'], errors='coerce')

        self.model = None
        self.inputs = inputs

    def build_nn(self, input_shape, output_shape, fc1_shape, fc2_shape, lr):
        self.model = Sequential([
            Dense(fc1_shape, input_shape=(input_shape, )),
            Activation('relu'),
            Dense(fc2_shape),
            Activation('relu'),
            Dense(output_shape)
        ])

        self.model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])

    def train_nn(self):
        X = self.db_features.values
        y = self.db_outputs.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)

        X_train = (X_train - mean) / std

        input_shape = X_train.shape[1]
        output_shape = y_train.shape[1]
        self.build_nn(input_shape, output_shape, 256, 256, 0.0001)

        model_filename = 'model.hdf5'
        model_chkpt = r"models/" + model_filename
        callbacks = [EarlyStopping(monitor='val_loss', patience=50),
                     ModelCheckpoint(model_chkpt, monitor='loss', save_best_only=True, mode='min')]
        history = self.model.fit(X_train, y_train, validation_split=0.1, epochs=100, callbacks=callbacks, batch_size=1)
        visualize_training(history=history)

    def test_nn(self):
        self.model = load_model(r"models/model.hdf5")

        new_test = {
            'fo': 3500000000,
            'bandwidth': 2e8,
            'length': 0.0158,
            'mclin': 5
        }

        self.db_features = self.db_features.append(new_test, ignore_index=True)

        X = self.db_features.values

        mean = X.mean(axis=0)
        std = X.std(axis=0)
        data_x = (X - mean) / std

        new_x = np.asarray([data_x[-1]])
        preditted = self.model.predict(new_x)
        print(preditted)

    def predict_nn(self):
        self.model = load_model(r"models/model.hdf5")
        self.db_features = self.db_features.append(self.inputs, ignore_index=True)

        X = self.db_features.values

        mean = X.mean(axis=0)
        std = X.std(axis=0)
        data_x = (X - mean) / std

        new_x = np.asarray([data_x[-1]])
        preditted = self.model.predict(new_x)
        print(preditted)
        return preditted


if __name__ == '__main__':
    dpath = r"datasets/mclin.xlsx"
    sample = {
        'fo': 3500000000,
        'bandwidth': 2e8,
        'length': 0.0158,
        'mclin': 4
    }
    nn = NN(dpath, sample)
    nn.predict_nn()
