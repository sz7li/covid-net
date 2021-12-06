import argparse
import os
import io
import json
import subprocess
import psutil

from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from matplotlib import pyplot as plt 
import numpy as np

from PIL import Image
from google.cloud import storage

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras import Input
from tensorflow.keras import regularizers

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/root/.config/gcloud/application_default_credentials.json"

LOCAL_PATH = "/tmp/keras-model"
process = psutil.Process(os.getpid())
print(f"GPUS {tf.config.list_physical_devices('GPU')}")

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, images_path, batch_size=32, dim=(512, 512), n_channels=3,
                 n_classes=1, shuffle=True, to_fit=True, images_in_mem=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.to_fit = to_fit
        self.images_path = images_path
        self.images_in_mem = images_in_mem

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        X = self._generate_X(list_IDs_temp)
        
        # Generate data
        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def _generate_X(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        if self.images_in_mem:
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = self.images_in_mem[ID]
        else:
            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                with tf.io.gfile.GFile(self.images_path + ID, 'rb') as fid: 
                    encoded_jpg = fid.read()
                encoded_jpg_io = io.BytesIO(encoded_jpg)
                image = Image.open(encoded_jpg_io)
                X[i,] = np.expand_dims(np.asarray(image.convert('L')) / 255.0, axis=-1)
        
        return X
            
    def _generate_y(self, list_IDs_temp):
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            # Store class
            y[i] = self.labels[ID]
        
        return y

def create_model():
    inputs = Input(shape=(512, 512, 1))

    conv = Conv2D(24, (3, 3), activation='relu') (inputs)
    conv = MaxPooling2D((2, 2)) (conv)
    conv = Dropout(0.2) (conv)

    # conv = Conv2D(32, (3, 3), activation='relu') (inputs)
    # conv = MaxPooling2D((2, 2)) (conv)
    # conv = Dropout(0.2) (conv)

    conv = Flatten()(conv)
    conv = Dense(64, activation="relu") (conv)
    conv = Dropout(0.2) (conv)

    conv = Dense(64, activation="relu") (conv)
    conv = Dropout(0.2) (conv)

    output = Dense(1, activation="sigmoid") (conv)

    model = Model(inputs=inputs, outputs=output)

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["mae", "accuracy"])
    print(model.summary())

    return model

def load_data(images_path, files):
    data = {}
    for ID in tqdm(files):
        with tf.io.gfile.GFile(images_path + ID, 'rb') as fid: 
            encoded_jpg = fid.read()

        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        data[ID] = np.expand_dims(np.asarray(image.convert('L')) / 255.0, axis=-1)

    return data

def _is_chief(cluster_resolver):
    task_type = cluster_resolver.task_type
    return task_type is None or task_type == 'chief'


def _get_temp_dir(model_path, cluster_resolver):
    worker_temp = f'worker{cluster_resolver.task_id}_temp'
    return os.path.join(model_path, worker_temp)


def save_model(model_path, model):
    # the following is need for TF 2.2. 2.3 onward, it can be accessed from strategy
    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    is_chief = _is_chief(cluster_resolver)

    if not is_chief:
        model_path = _get_temp_dir(model_path, cluster_resolver)

    model.save(model_path)

    if is_chief:
        # wait for workers to delete; check every 100ms
        # if chief is finished, the training is done
        while tf.io.gfile.glob(os.path.join(model_path, "worker*")):
            sleep(0.1)

    if not is_chief:
        tf.io.gfile.rmtree(model_path)

if __name__ == "__main__":
    
    # data_path = "gs://covid-net/train/"
    # data_labels_path = "gs://covid-net/train_labels.csv"

    # tf_config_str = os.environ.get('TF_CONFIG')
    # tf_config_dict  = json.loads(tf_config_str)

    # print(json.dumps(tf_config_dict, indent=2))
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-files',
        nargs='+',
        help='Training file local or GCS',
        default=[
            'gs://covid-net/train/'])
    parser.add_argument(
        '--train-indices',
        nargs='+',
        help='train indices',
        default=[
            "gs://covid-net/train_indices.csv"])
    parser.add_argument(
        '--val-indices',
        nargs='+',
        help='validation indices',
        default=[
            "gs://covid-net/val_indices.csv"])
    parser.add_argument(
        '--data-labels',
        nargs='+',
        help='data labels by ID',
        default=[
            "gs://covid-net/train_labels.csv"])
    parser.add_argument(
        '--job-dir',
        nargs='+',
        help='job directory',
        default=[
            "gs://covid-net/models/"])

    # data_path = os.path.join(os.getcwd(), '../train/train/')
    # data_labels_path = '../train_labels.csv'

    args, _ = parser.parse_known_args()
    arguments = args.__dict__

    with tf.io.gfile.GFile(arguments['data_labels'][0], 'rb') as fid:
        data_labels = fid.read()
    
    df_true = pd.read_csv(io.BytesIO(data_labels)).head(250)

    print(df_true.head())

    x_train_indices, x_val_indices, y_train_indices, y_val_indices = train_test_split(df_true['File'], df_true['Label'], test_size=0.01, random_state=441)
    partition = {}
    partition['train'] = x_train_indices.values
    partition['validation'] = x_val_indices.values
    labels = {v['File']:v['Label'] for (k, v) in df_true.transpose().to_dict().items()}

    print(f"Loading data from {arguments['train_files'][0]}")
    data = load_data(arguments['train_files'][0], df_true['File'])

    # Parameters
    params = {'dim': (512, 512),
            'batch_size': 32,
            'n_classes': 1,
            'n_channels': 1,
            'shuffle': True,
            'images_in_mem': data}

    # Distributed strategy
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    nr_of_replicas = strategy.num_replicas_in_sync
    global_batch_size = nr_of_replicas * params['batch_size']

    params['batch_size'] = global_batch_size

    print(f"Nr of replicas: {nr_of_replicas} global batch size :{global_batch_size}, data total size: {df_true.shape[0]}")

    print(f"Creating model, mem usage: {process.memory_info().rss}")

    with strategy.scope():
        print("Creating model with training strategy")
        # Create Model
        multi_worker_model = create_model()

    print(f"Model created, mem usage {process.memory_info().rss}")

    # Generators
    training_generator = DataGenerator(partition['train'], labels, arguments['train_files'][0], **params)
    # validation_generator = DataGenerator(partition['validation'], labels, data_path, **params)

    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(df_true['Label']), y=df_true['Label'])
    class_weights = {0: class_weights[0], 1: class_weights[1]}

    print("Fitting model")

    NUM_EPOCHS = 3

    # Train model on dataset
    multi_worker_model.fit(
        x=training_generator, 
        epochs=NUM_EPOCHS, 
        steps_per_epoch=np.ceil(df_true.shape[0] / params['batch_size']),
        class_weight=class_weights,
        max_queue_size=20,
        workers=8,
        # use_multiprocessing=True,
    )

    print("Saving model")

    # save_model(arguments['job_dir']['0'], multi_worker_model)

    print(f"Process finished, mem usage {process.memory_info().rss}")

    # with tf.io.gfile.GFile(arguments['data_labels'][0], 'rb') as fid: 
    #     data_labels = fid.read()

    # model.save('gs://covid-net/')


