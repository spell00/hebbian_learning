'''Trains a simple deep NN on the dementia dataset.
total of 624 samples:
312 alzheimer
156 controls
156 hungtington's disease


'''

from __future__ import print_function
import numpy as np


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout,BatchNormalization
from keras.optimizers import Adam
from sklearn.preprocessing import RobustScaler
import random
import matplotlib.pyplot as plt
import operator

batch_size = 128
epochs = 100
nrep = 10
init = "he_uniform"

# the data, shuffled and split between train and test sets
data1 = np.load('dementia_gene_expression.pickle.npy')
data1[np.isnan(data1)] = 0
labels = np.load('dementia_labels.pickle.npy')
# Replace labels with 0,1,2 for not demented, alzheimer and hungtington's disease, respectively
controlHuntington = labels[labels != " Alzheimer's disease"]
controlAlzheimer = labels[labels != " Huntington's disease"]
huntingtonAlzheimer = labels[labels != ' non-demented']
data = data1[:,labels != " Huntington's disease"]
filenames = ['controlHuntington_history','controlAlzheimer_history','huntingtonAlzheimer_history','controlHuntingtonAlzheimer_history','controlDisease_history']
Labels = [controlHuntington,controlAlzheimer,huntingtonAlzheimer,labels,labels]
dataBinary = [data1[:,labels != " Alzheimer's disease"],data1[:,labels != " Huntington's disease"],data1[:,labels != ' non-demented']]
for l in range(len(Labels)):
    accuracy_training_array = []
    accuracy_valid_array = []
    losses_training_array = []
    losses_valid_array = []
    max_valid_accuracies = []
    max_valid_epochs = []
    min_valid_loss = []
    min_valid_loss_epochs = []
    for rep in range(nrep):
        print("Repetition: " , str(rep+1), "/", nrep)
        csv_logger = keras.callbacks.CSVLogger('dementia_mlp3_'+filenames[l]+'.csv')
        labels = Labels[l]
        label_set = set(labels)
        if(l < len(dataBinary)):
            num_classes = 2
            data = dataBinary[l]

        else:
            data = data1
            num_classes = 3
            label_set = set(labels)

        classes = np.copy(labels)
        if(l == len(Labels)+1):
            num_classes = 2
            for lab in range(len(labels)):
                if (labels[lab] == ' non-demented'):
                    classes[lab] = 0
                else:
                    classes[lab] = 1
        else:
            for index, label in enumerate(label_set):
                for lab in range(len(labels)):
                    if (label == labels[lab]):
                        classes[lab] = int(index)

        classes = np.array(classes)
        random_training = np.random.choice(range(len(classes)),size=(int(len(classes)*(5/6))),replace=False)
        x_train = data[:,random_training]
        x_test = np.delete(data,random_training,axis=1)
        y_train = classes[random_training]
        y_test = np.array(np.delete(classes,random_training))


        #x_train *= 1000000000
        #x_test 	*= 1000000000
        x_train = np.transpose(x_train)
        x_test = np.transpose(x_test)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        model = Sequential()

        model.add(Dense(1024, activation='relu',kernel_initializer=init ,kernel_regularizer=keras.regularizers.l1_l2(0.0001,0.0001), input_shape=(x_train.shape[1],)))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(1024, activation='relu',kernel_initializer=init ,kernel_regularizer=keras.regularizers.l1_l2(0.0001,0.0001)))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(1024, activation='relu',kernel_initializer=init ,kernel_regularizer=keras.regularizers.l1_l2(0.0001,0.0001)))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(num_classes, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_test, y_test),
                            callbacks=[csv_logger])
        accuracy_training_array.append(history.history['acc'])
        accuracy_valid_array.append(history.history['val_acc'])

        maxepoch, maxvalid = max(enumerate(history.history['val_acc']), key=operator.itemgetter(1))
        minlossepoch, minloss = min(enumerate(history.history['val_loss']), key=operator.itemgetter(1))
        max_valid_accuracies.append(maxvalid)
        max_valid_epochs.append(maxepoch)
        min_valid_loss.append(minloss)
        min_valid_loss_epochs.append(minlossepoch)
        losses_training_array.append(history.history['loss'])
        losses_valid_array.append(history.history['val_loss'])

    accuracy_training_means = [np.mean(x) for x in np.transpose(accuracy_training_array)]
    accuracy_training_std = [np.std(x) for x in np.transpose(accuracy_training_array)]
    accuracy_training_sem = np.array(accuracy_training_std) / [np.sqrt(len(x_test))]

    accuracy_valid_means = [np.mean(x) for x in np.transpose(accuracy_valid_array)]
    accuracy_valid_std = [np.std(x) for x in np.transpose(accuracy_valid_array)]
    accuracy_valid_sem = np.array(accuracy_valid_std) / [np.sqrt(len(x_test))]

    losses_training_means = [np.mean(x) for x in np.transpose(losses_training_array)]
    losses_training_std = [np.std(x) for x in np.transpose(losses_training_array)]
    losses_training_sem = np.array(losses_training_std) / [np.sqrt(len(x_test))]

    losses_valid_means = [np.mean(x) for x in np.transpose(losses_valid_array)]
    losses_valid_std = [np.std(x) for x in np.transpose(losses_valid_array)]
    losses_valid_sem = np.array(losses_valid_std) / [np.sqrt(len(x_test))]


    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


