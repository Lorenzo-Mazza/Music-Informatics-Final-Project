import pickle
import random

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import os
import pandas as pd
import numpy as np
import madmom
from tensorflow.python.keras.layers import Conv1D
from tensorflow.keras.layers import GlobalAveragePooling1D, GaussianNoise
from wrn import *
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import *

l1_l2 = tf.keras.regularizers.l1_l2


class WarmUpPiecewiseConstantSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule used in the original paper
    """

    def __init__(self,
                 steps_per_epoch,
                 base_learning_rate,
                 decay_ratio,
                 decay_epochs,
                 warmup_epochs):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.base_learning_rate = base_learning_rate
        self.decay_ratio = decay_ratio
        self.decay_epochs = decay_epochs
        self.warmup_epochs = warmup_epochs

    def __call__(self, step):
        lr_epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
        learning_rate = self.base_learning_rate
        if self.warmup_epochs >= 1:
            learning_rate *= lr_epoch / self.warmup_epochs
        decay_epochs = [self.warmup_epochs] + self.decay_epochs
        for index, start_epoch in enumerate(decay_epochs):
            learning_rate = tf.where(
                lr_epoch >= start_epoch,
                self.base_learning_rate * self.decay_ratio ** index,
                learning_rate)
        return learning_rate

    def get_config(self):
        return {
            'steps_per_epoch': self.steps_per_epoch,
            'base_learning_rate': self.base_learning_rate,
        }


RUN_ID = '3002'   # yet to run,   3 _ _ _ IS MGT DATASET
SECTION = 'NOWRN'
PARENT_FOLDER = os.getcwd()
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join(RUN_ID)
if not os.path.exists(RUN_FOLDER):
    os.makedirs(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))
    os.mkdir(os.path.join(RUN_FOLDER, 'metrics'))
    os.mkdir(os.path.join(RUN_FOLDER, 'plots'))


def build_cnn(input_dims, l1, l2):
    inputs = Input(shape=(input_dims))
    x = tf.keras.layers.Reshape(input_dims + (1,))(inputs)
    kernel_regularizer = l1_l2(l1=l1, l2=l2)
    #x= GaussianNoise(0.1)(x)
    x = Conv2D(filters=4, kernel_size=5, padding='same', activation='elu', kernel_regularizer=kernel_regularizer)(x)
    x = Conv2D(filters=4, kernel_size=5, padding='same', activation='elu', kernel_regularizer=kernel_regularizer)(x)
    x = Conv2D(filters=4, kernel_size=5, padding='same', activation='elu', kernel_regularizer=kernel_regularizer)(x)
    x = Conv2D(filters=8, kernel_size=5, padding='same', activation='elu', kernel_regularizer=kernel_regularizer)(x)
    x = Conv2D(filters=8, kernel_size=5, padding='same', activation='elu', kernel_regularizer=kernel_regularizer)(x)
    x = tf.keras.layers.Reshape((input_dims[0], -1))(x)
    x = Dense(units=48, activation='elu')(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(24, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)



with (open('mgt30sec.pickle',"rb")) as f:
    x,y = pickle.load(f)
a= np.max(x)
b= np.min(x)
Nx = len(x)
noise_factor=0.1
noise= noise_factor* np.random.randn(Nx,*x[0].shape)
#x+= noise
temp = list(zip(x,y))
random.shuffle(temp)
x,y = zip(*temp)


epochs = 500

learning_rate = 0.005  # 0.005
l1_reg = .001
l2_reg = .001 # l2>= 0.05 breaks training
training_batchsize = 64
validation_batchsize = 64
split = 0.3

decay_epochs = [10, 20, 40, 180]
lr_decay_epochs = [(int(start_epoch_str) * epochs) // 800 for start_epoch_str in decay_epochs]
lr_decay_ratio = 0.2
lr_warmup_epochs = 1


network = build_cnn((12, 150), l1=l1_reg, l2=l2_reg)

steps_per_epoch = Nx // training_batchsize
lr_schedule = WarmUpPiecewiseConstantSchedule(
    steps_per_epoch,
    learning_rate,
    decay_ratio=lr_decay_ratio,
    decay_epochs=lr_decay_epochs,
    warmup_epochs=lr_warmup_epochs)
#optimizer = tf.keras.optimizers.SGD(lr_schedule, momentum=0.9, nesterov=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, nesterov=True)
#optimizer = Adam(learning_rate=learning_rate)
label_smoothing = 0.0
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
# Prepare the metrics.
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
network.compile(
    optimizer=optimizer,  # Optimizer
    # Loss function to minimize
    loss=loss_fn,
    # List of metrics to monitor
    metrics=['accuracy']
)

lines = ["n_epochs= " + str(epochs), "decay epochs= " + str(decay_epochs),"decay ratio= " + str(lr_decay_ratio),
         "lr= " + str(learning_rate), "label_smoothing= " + str(label_smoothing),
         "l1 reg= " + str(l1_reg), "l2 reg= " + str(l2_reg),
         "tr batch= " + str(training_batchsize), "val batch= " + str(validation_batchsize),
         "validation split= " + str(split)]
with open(os.path.join(RUN_FOLDER, 'model_params.txt'), 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')

checkpoint_filepath = os.path.join(RUN_FOLDER, 'weights/final_weights.h5')
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)



history = network.fit(x=np.array(x, dtype=float), y=np.array(y, dtype=int),
                      epochs=epochs, batch_size=training_batchsize, validation_batch_size=validation_batchsize,
                      validation_split=split, callbacks=[EarlyStopping(patience=100),model_checkpoint_callback],
                      shuffle=True)

with open(os.path.join(RUN_FOLDER, 'metrics/metrics_evo.pickle'), 'wb') as f:
    pickle.dump(history.history, f)


from matplotlib.ticker import MaxNLocator

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(history.history['loss'], label='Categorical Cross-Entropy (training data)')
plt.plot(history.history['val_loss'], label='Categorical Cross-Entropy (validation data)')
plt.title('Loss function - WRN 28-2')
plt.ylabel('Categorical Cross-Entropy')
plt.xlabel('No. epoch')
plt.legend()
plt.savefig(os.path.join(RUN_FOLDER, 'plots/loss_plot.png'))
plt.show()

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Accuracy - WRN 28-2')
plt.ylabel('Accuracy')
plt.xlabel('No. epoch')
plt.legend()
plt.savefig(os.path.join(RUN_FOLDER, 'plots/accuracy_plot.png'))
plt.show()

