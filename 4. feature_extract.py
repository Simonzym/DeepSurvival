# load required libraries

import numpy as np
import pandas as pd
import tensorflow as tf
import rpy2.robjects as robjects
import pyreadr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
os.environ['KERAS_BACKEND'] = 'theano'
import theano.tensor as T
import keras.backend as K
from keras.applications import inception_v3
from keras.applications.inception_v3 import preprocess_input
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter

#the path to the clinical covariates and slices
image_path = 'slice'
#data_path = '/Users/Menghan/Google Drive/Brown/Year5/proj3/data'
features = pd.read_csv(os.path.join('data431_surv.csv'), sep = ',', header = 0)

# rename columns to avoid dot
features = features.rename(columns = {'Overall.Stage': 'Overall_Stage',
                           'clinical.T.Stage': 'Clinical_T_Stage',
                           'Clinical.N.Stage': 'Clinical_N_Stage',
                           'Clinical.M.Stage': 'Clinical_M_Stage',
                           'Survival.time': 'Survival_time',
                           'deadstatus.event': 'deadstatus_event'
                           })
features['shortID'] = features.PatientID.str.slice(6, 9)


count = 0
for col in features.columns:
    print(count)
    count += 1


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

for i in range(N):
    if (i % 10 == 0):
        print(i)
    try:
        image = cv2.imread(os.path.join(image_path, image_names[i]))
        if (image.shape == (218, 218, 3)):
            x.append(image)
            index = np.where(features.shortID == image_ids[i])[0][0]
            ft.append(features.iloc[index, feature_idx])
            y.append(time[index])
            e.append(event[index])
    except:
        continue

inception_model = inception_v3.InceptionV3(include_top = False, weights = 'imagenet', input_shape = (218, 218, 3))
x2 = []
for i in range(len(x)):
    if (i % 10 == 0):
        print(i)
    img = x[i]
    img = preprocess_input(img.reshape(1, 218, 218, 3))
    img_new = inception_model.predict(img.reshape(1, 218, 218, 3))
    x2.append(img_new)

x2 = np.array(x2)
# x2 = np.concatenate(x2, axis = 0)
x2.shape
x2 = x2.reshape(x2.shape[0], x2.shape[2], x2.shape[3], x2.shape[4])

'''
X_train, X_val, Y_train, Y_val = train_test_split(x2, y, test_size=0.2, random_state=10)
X_train, X_val, E_train, E_val = train_test_split(x2, e, test_size=0.2, random_state=10)
'''


def negative_log_likelihood(E):
    def loss(y_true, y_pred):
        hazard_ratio = K.exp(y_pred)
        log_risk = K.log(K.extra_ops.cumsum(hazard_ratio))
        uncensored_likelihood = y_pred.K - log_risk
        censored_likelihood = uncensored_likelihood * E
        neg_likelihood = -K.sum(censored_likelihood)
        return neg_likelihood
    return loss


inception_top = Sequential()
inception_top.add(Conv2D(512, kernel_size = (3,3),
                         activation = 'relu',
                         input_shape = (x2.shape[1], x2.shape[2], x2.shape[3])))
inception_top.add(MaxPooling2D(pool_size = (2,2)))
inception_top.add(Flatten())
inception_top.add(Dense(256, activation = 'relu'))
inception_top.add(Dropout(0.5))
inception_top.add(Dense(256, activation = 'relu'))
inception_top.add(Dropout(0.5))
inception_top.add(Dense(1, activation = 'linear', kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.01)))
sgd = SGD(lr=1e-5, decay=0.01, momentum=0.9, nesterov=True)
inception_top.compile(loss=negative_log_likelihood(e), optimizer=sgd)
# inception_top.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history_inception = inception_top.fit(x2/np.max(x2), np.asarray(e), batch_size=285, epochs=1000, shuffle = False)

'''
hr_pred = inception_top.predict(X_train)
hr_pred = np.exp(hr_pred)
ci=concordance_index(Y_train,-hr_pred,E_train)

hr_pred2 = inception_top.predict(X_val)
hr_pred2 = np.exp(hr_pred2)
ci2 = concordance_index(Y_val,-hr_pred2,E_val)
print('Concordance Index for training dataset:', ci)
print('Concordance Index for test dataset:', ci2)
'''

inception_top.summary()

mid_layer = Model(inputs = inception_top.inputs, outputs = inception_top.get_layer('conv2d_100').output)
conv2d_feature = mid_layer.predict(x2)

mid_layer = Model(inputs = inception_top.inputs, outputs = inception_top.get_layer('dense_11').output)
dense11_feature = mid_layer.predict(x2)


# add column names and change data types
ft = pd.DataFrame(ft)

names = ['age', 'Clinical_T_Stage', 'Clinical_N_Stage', 'Clinical_M_Stage', 'Overall_Stage', 'Histology', 'gender'] + \
        ['rad_' + str(int(i)) for i in range(1, ft.shape[1] - 6)]
ft.columns = names

ft[['age', 'Clinical_T_Stage', 'Clinical_N_Stage', 'Clinical_M_Stage'] + ['rad_' + str(int(i)) for i in range(1, ft.shape[1] - 6)]] = \
    ft[['age', 'Clinical_T_Stage', 'Clinical_N_Stage', 'Clinical_M_Stage'] + ['rad_' + str(int(i)) for i in range(1, ft.shape[1] - 6)]].apply(pd.to_numeric)
ft[['Overall_Stage', 'Histology', 'gender']] = \
    ft[['Overall_Stage', 'Histology', 'gender']].astype(str)

# categorical features
ft_cat = ft[['Overall_Stage', 'Histology', 'gender']]

# label encode categorical variables with strings
le_overallstage = LabelEncoder()
le_Histology = LabelEncoder()
le_gender = LabelEncoder()
ft_cat['stage_encoded'] = le_overallstage.fit_transform(ft_cat.Overall_Stage)
ft_cat['histology_encoded'] = le_Histology.fit_transform(ft_cat.Histology)
ft_cat['gender_encoded'] = le_gender.fit_transform(ft_cat.gender)

# one-hot-encoding
ohe_overallstage = OneHotEncoder()
ohe_Histology = OneHotEncoder()
ohe_gender = OneHotEncoder()

overallstage = ohe_overallstage.fit_transform(ft_cat.stage_encoded.values.reshape(-1, 1)).toarray()
overallstage = pd.DataFrame(overallstage, columns = ["Stage_" + str(int(i)) for i in range(overallstage.shape[1])])

Histology = ohe_Histology.fit_transform(ft_cat.histology_encoded.values.reshape(-1, 1)).toarray()
Histology = pd.DataFrame(Histology, columns = ["Histology_" + str(int(i)) for i in range(Histology.shape[1])])

gender = ohe_gender.fit_transform(ft_cat.gender_encoded.values.reshape(-1, 1)).toarray()
gender = pd.DataFrame(gender, columns = ["female", "male"])

# create feature dataframes
ft_clinical = ft[['age', 'Clinical_T_Stage', 'Clinical_N_Stage', 'Clinical_M_Stage']]
ft_clinical = pd.concat([ft_clinical.reset_index(drop=True),
                          overallstage.reset_index(drop=True),
                          Histology.reset_index(drop=True),
                          gender.reset_index(drop=True)],
                         axis = 1)
ft_rad = ft.drop(['age', 'Clinical_T_Stage', 'Clinical_N_Stage', 'Clinical_M_Stage', 'Overall_Stage', 'Histology', 'gender'],
                 axis = 1)
ft_inception = pd.DataFrame(data = dense11_feature, columns = ["dl_" + str(int(i)) for i in range(1, dense11_feature.shape[1] + 1)])
ft_total = pd.concat([ft_clinical.reset_index(drop = True),
                      ft_rad.reset_index(drop = True),
                      ft_inception.reset_index(drop = True)], axis = 1)
ft_total['Survival_time'] = y
ft_total['Status'] = e
ft_total.to_csv(os.path.join(data_path,'ft_total.csv'), sep=',', header=True, index = False)