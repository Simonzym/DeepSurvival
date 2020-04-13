# import libraries
import tensorflow as tf 
import numpy as np
import pandas as pd

import os
from keras import applications
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import optimizers
from keras.optimizers import SGD, RMSprop, Adam
from keras.regularizers import l2

from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
import h5py
import nibabel as nib
import theano
import theano.tensor as tt
import keras.backend as K

from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter

data_path = ''
image_path = 'box_sq'
features = pd.read_csv(os.path.join(data_path, 'data431_surv.csv'), sep = ',', header = 0)

# img = nib.load(os.path.join(image_path, 'LUNG1-001-box.nii.gz'))
# rename columns to avoid dot
features = features.rename(columns = {'Overall.Stage': 'Overall_Stage',
                           'clinical.T.Stage': 'Clinical_T_Stage',
                           'Clinical.N.Stage': 'Clinical_N_Stage',
                           'Clinical.M.Stage': 'Clinical_M_Stage',
                           'Survival.time': 'Survival_time',
                           'deadstatus.event': 'deadstatus_event'
                           })
features['shortID'] = features.PatientID.str.slice(6, 9)

feature_idx = list(range(1, 8)) + list(range(10, 441))
features[['Survival_time']] = features[['Survival_time']].astype(float)
time = features['Survival_time']
event = features['deadstatus_event']

x = []
ft = []
y = []
e = []

image_names = [f for f in os.listdir(image_path) if not f.startswith('.')]
image_ids = [f[6:9] for f in image_names]
N = len(image_names)

exclude = [7, 34, 36, 40, 44, 50, 58, 67, 68, 83, 84, 94, 95, 96, 110, 128,
           135, 137, 143, 144, 146, 164, 166, 167, 176, 191, 200, 202, 208,
           212, 214, 222, 227, 230, 238, 250, 251, 254, 265, 279, 301, 302,
           312, 314, 316, 319, 322, 324, 325, 327, 329, 330, 333, 336, 338,
           344, 346, 348, 350, 351, 352, 354, 355, 356, 357, 358, 359, 360,
           361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 373, 374,
           375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387,
           388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400,
           401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413,
           414, 415, 416, 417, 418, 419, 420, 421, 422]
exclude_char = [format(x, '03d') for x in exclude]
ids = []
for i in range(N):
    if (i % 10 == 0):
        print(i)
    try:
        if (image_ids[i] not in set(exclude_char)):
            image = nib.load(os.path.join(image_path, image_names[i]))
            if (image.shape == (218, 218, 58)):
                x.append(image.get_fdata())
                print("id is", image_ids[i])
                ids.append(image_ids[i])
                index = np.where(features.shortID == image_ids[i])[0][0]
                ft.append(features.iloc[index, feature_idx])
                y.append(time[index])
                print("time is", time[index])
                e.append(event[index])
    except:
        continue

x = np.array(x)
x = x/(np.max(x) - np.min(x))
e = np.array(e)
y = np.array(y)


def negative_log_likelihood(E):
    def loss(y_true, y_pred):        
        hazard_ratio = K.exp(y_pred)
        log_risk = K.log(K.cumsum(hazard_ratio))
        uncensored_likelihood = y_pred - log_risk
        censored_likelihood = uncensored_likelihood * E
        neg_likelihood = -K.sum(censored_likelihood)
        return neg_likelihood
    return loss

all_id = list(range(0, len(ids)))
folds = pd.read_csv('index_mat.csv', index_col = 0).to_numpy() - 1
all_pred = []
for j in range(15):
    
    id_use = [i for i in all_id if i not in folds[j,]]

    x_use = x[id_use]
    y_use = y[id_use]
    e_use = e[id_use]

    nn_3D = Sequential()
    nn_3D.add(Conv2D(64, kernel_size = (3,3),
                         activation = 'relu', 
                         input_shape = (x.shape[1], x.shape[2], x.shape[3])))
    nn_3D.add(MaxPooling2D(pool_size = (2,2)))
    nn_3D.add(Flatten())
    nn_3D.add(Dense(256, activation = 'relu'))
    nn_3D.add(Dropout(0.5))
    nn_3D.add(Dense(256, activation = 'relu'))
    nn_3D.add(Dropout(0.5))
    nn_3D.add(Dense(1, activation = 'linear', kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.01)))
    sgd = SGD(lr=1e-6)
    #rmsprop=RMSprop(lr=1e-5, rho=0.9, epsilon=1e-8)
    nn_3D.compile(loss=negative_log_likelihood(e), optimizer=sgd)
    nn_3D.fit(x, y, batch_size = 19, epochs = 10, shuffle = False)

    hr_pred = nn_3D.predict(x)
    all_pred.append(hr_pred)
    
cindex = []
for j in range(15):
    
    id_use = [i for i in all_id if i in folds[j,]]
    
    hr = np.exp(pred_out[j,id_use])
    ci = concordance_index(y[id_use], -hr, e[id_use])
    cindex.append(ci)

pred_out = np.array(all_pred).reshape(15, 285)
np.savetxt('hr.csv', pred_out)    
ids_np = np.array(ids).astype('float32')
np.savetxt('ids.csv', ids_np)

