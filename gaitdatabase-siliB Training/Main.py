# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:44:59 2020

@author: Mahdi
"""
#%% Library Importing and Data Loading


import os
import shutil
import pickle
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob

################################## REAL TEST ###################################

# #%% If data is not prepared yet :
        
# datadir = 'E:\\My Projects\\2019 Works\\Datasets\\For Computer Vision\\GaitDatasetB-silh'

# os.chdir(datadir)

# if 'Train' and 'Test' not in os.listdir():
#     os.mkdir('Train')
#     os.mkdir('Test')
#     os.mkdir('Validation')


# Val_folderer = np.random.permutation(range(1,125))
# valsize = round(0.33*124/2)
# testsize = round(0.2*(124 - valsize))

# val_folders = Val_folderer[:valsize]
# test_folders = Val_folderer[valsize:valsize + testsize]


# for i in np.sort(val_folders):
#     if i<10:
#          i = '00' + str(i)
#     elif i< 100:
#         i = '0' + str(i)
#     src = datadir + '\\' + str(i)
#     dest = datadir + '\\Validation'
#     shutil.move(src , dest)
    


# for i in np.sort(test_folders):
#     if i<10:
#          i = '00' + str(i)
#     elif i< 100:
#         i = '0' + str(i)
#     src = datadir + '\\' + str(i)
#     dest = datadir + '\\Test'
#     shutil.move(src , dest)
    

    
# #%% Data Preparing II:
# os.chdir(datadir)

# os.chdir(datadir + '\\Test')    
# for i in glob.glob('*/*/*/*'):
#     tmp = i.split(sep = '\\')[3]
#     # print(i)
#     # print(tmp)
#     tmp2 = tmp.split(sep = '-')
#     if tmp2[0] in os.listdir():
#         shutil.move(datadir + '\\Test' +'\\' + i , datadir + '\\Test' + '\\' + tmp2[0])

# os.chdir(datadir + '\\Test')    
# for i in os.listdir():
#     os.chdir(datadir +'\\Test\\' + i)
#     for j in os.listdir():
#         if not j.endswith('.png'):
#             # print(j)
#             shutil.rmtree(j)
            
            
# os.chdir(datadir + '\\Validation')    
# for i in glob.glob('*/*/*/*'):
#     tmp = i.split(sep = '\\')[3]
#     # print(i)
#     # print(tmp)
#     tmp2 = tmp.split(sep = '-')
#     if tmp2[0] in os.listdir():
#         shutil.move(datadir + '\\Validation' +'\\' + i , datadir + '\\Validation' + '\\' + tmp2[0])

# os.chdir(datadir + '\\Validation')    
# for i in os.listdir():
#     os.chdir(datadir +'\\Validation\\' + i)
#     for j in os.listdir():
#         if not j.endswith('.png'):
#             # print(j)
#             shutil.rmtree(j)
            

# os.chdir(datadir + '\\Train')    
# for i in glob.glob('*/*/*/*'):
#     tmp = i.split(sep = '\\')[3]
#     # print(i)
#     # print(tmp)
#     tmp2 = tmp.split(sep = '-')
#     if tmp2[0] in os.listdir():
#         shutil.move(datadir + '\\Train' +'\\' + i , datadir + '\\Train' + '\\' + tmp2[0])

# os.chdir(datadir + '\\Train')    
# for i in os.listdir():
#     os.chdir(datadir +'\\Train\\' + i)
#     for j in os.listdir():
#         if not j.endswith('.png'):
#             # print(j)
#             shutil.rmtree(j)
            
# #%% Creating Method Folders:

# start = time.time()
# os.chdir(datadir)    
# shutil.copytree(datadir+'\\Train' , datadir+'\\Train 1')
# shutil.copytree(datadir+'\\Train' , datadir+'\\Train 2')
# shutil.copytree(datadir+'\\Train' , datadir+'\\Train 3')
# shutil.copytree(datadir+'\\Train' , datadir+'\\Train 4')
# shutil.copytree(datadir+'\\Validation' , datadir+'\\Val 1')
# shutil.copytree(datadir+'\\Validation' , datadir+'\\Val 2')
# shutil.copytree(datadir+'\\Validation' , datadir+'\\Val 3')
# shutil.copytree(datadir+'\\Validation' , datadir+'\\Val 4')
# shutil.copytree(datadir+'\\Test' , datadir+'\\Test 1')
# shutil.copytree(datadir+'\\Test' , datadir+'\\Test 2')
# shutil.copytree(datadir+'\\Test' , datadir+'\\Test 3')
# shutil.copytree(datadir+'\\Test' , datadir+'\\Test 4')
# end = time.time()

# print(f'Elapsed Time For Copying : \n{end-start} seconds == \n{(end-start)/60} minutes == \n{(end-start)/3600} hours')


#%% Preparing datas in Test, Validation and Train for Human Identification 

# os.chdir(datadir + '\\Test')
# for i in os.listdir():
#     src = datadir + '\\Test\\' + str(i)
#     dest = datadir + '\\Train'
#     shutil.move(src , dest)

datadir = 'E:\\My Projects\\2019 Works\\Datasets\\For Computer Vision\\GaitDatasetB-silh\\Dataset1'

os.chdir(datadir)

### We wanna split one person's pics into test,train,val parts and, for 
#   classification, we will have a one hot encoded vector of length 124 for 124
#   persons; in which every person that the data belongs to, have a 1 and others
#   are zero.
## Finding the suitable test train val split for each person

# # Specifying validation and test size for each person
# os.chdir(datadir)
# Val_folderer = []
# valsize = []
# testsize = []
# counter = 0
# for i in os.listdir():
#     os.chdir(datadir + '\\' + i)
#     tmp = len(os.listdir())
#     valsizetmp = round(0.2*tmp)
#     valsize.append(valsizetmp)
#     testsize.append(round(0.2*(tmp - valsizetmp)))

# # # Doing folder settings
# os.chdir(datadir)
# if 'Train' and 'Test' and 'Valid' not in os.listdir():
#     os.mkdir('Train')
#     os.mkdir('Test')
#     os.mkdir('Valid')

# # # Split train, test, valid for persopns
# container = ['Train' , 'Test' , 'Valid']
lister = [str(i) for i in np.arange(1,125)]
# for i in container:
#     os.chdir(datadir + '\\' + i)
#     for j in lister:
#         if j not in os.listdir():
#             os.mkdir(j)

# os.chdir(datadir)    
# count = 0
# counter = 0
# for i in os.listdir():
#     if i != 'Train' and i != 'Test' and i != 'Valid':
#         os.chdir(datadir + '\\' + i)
#         for j in glob.glob('*'):
#             if counter <= valsize[count]:
#                 shutil.move(datadir + '\\' + i + '\\' + j , datadir + '\\Valid' + '\\' + lister[count])
#                 counter += 1
#             else:
#                 counter = 0
#                 break
            
#         for z in glob.glob('*'):
#             if counter <= testsize[count]:
#                 shutil.move(datadir + '\\' + i + '\\' + z , datadir + '\\Test' + '\\' + lister[count])
#                 counter += 1
#             else:
#                 counter = 0
#                 count +=1
#                 break
            
os.chdir(datadir)    
count = 0
for i in os.listdir():
    if i != 'Train' and i != 'Test' and i != 'Valid':
        os.chdir(datadir + '\\' + i)        
        for j in glob.glob('*'):
            shutil.move(datadir + '\\' + i + '\\' + j , datadir + '\\Train' + '\\' + lister[count]) 
        count += 1



#%% Shift Delete
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

#%% Libs

import shutil
import pickle
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D , Softmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob
import time
import PIL
from PIL import Image
from pathlib import Path
import pickle
# For Transfer Learning
# import tensorflow_hub as hub

#%% GPU Testing

tf.__version__

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)

tf.config.list_physical_devices('GPU')

tf.test.gpu_device_name()
            
#%% to use GPU or CPU use :

# with tf.device('/GPU:0'):

# with tf.device('/CPU:0'):
#%% Dataset I GPU

with tf.device('/GPU:0'):
    T0 = time.time()
    datadir = 'E:\\My Projects\\2019 Works\\Datasets\\For Computer Vision\\GaitDatasetB-silh\\Dataset1'
    
    os.chdir(datadir)
    
    traindir = datadir + '\\Train'
    validdir = datadir + '\\Valid'
    testdir = datadir + '\\Test'                        
    
    train_img_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255)
    valid_img_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255)
    test_img_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255)
    
    batch_size = 10
    epochs = 2
    img_height= 124     # 224 is the dimension for ImageNet
    img_width= 124
    
    
    # class_mode is for labeling... can be sparse(int), categorical(2D) , binary(1D) , None(for autoencoder) --> more info: keras.io
    train_data_gen0 = train_img_generator.flow_from_directory(batch_size = batch_size , directory = traindir , shuffle = True , target_size = (img_height , img_width) , class_mode = 'categorical')
    valid_data_gen0 = valid_img_generator.flow_from_directory(batch_size = batch_size , directory = validdir , shuffle = False , target_size = (img_height , img_width) , class_mode = 'categorical')
    test_data_gen0 = test_img_generator.flow_from_directory(batch_size = batch_size , directory = testdir , shuffle = False , target_size = (img_height , img_width) , class_mode = 'categorical')
    
    for img_batch , label_batch in train_data_gen0:
        print(f'Image_Batch_Size : {img_batch.shape}')
        print(f'Label_Batch_Size : {label_batch.shape}')
        break
    T1 = time.time()

print(f'Time for loading data on GPU:\n{T1-T0} seconds')

#%% Dataset I CPU

# with tf.device('/cpu:0'):
#     T2 = time.time()
#     datadir = 'E:\\My Projects\\2019 Works\\Datasets\\For Computer Vision\\GaitDatasetB-silh\\Dataset1'
    
#     os.chdir(datadir)
    
#     traindir = datadir + '\\Train'
#     validdir = datadir + '\\Valid'
#     testdir = datadir + '\\Test'                        
    
#     train_img_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255)
#     valid_img_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255)
#     test_img_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255)
    
#     batch_size = 10
#     epochs = 2
#     img_height= 124     # 224 is the dimension for ImageNet
#     img_width= 124
    
    
#     # class_mode is for labeling... can be sparse(int), categorical(2D) , binary(1D) , None(for autoencoder) --> more info: keras.io
#     train_data_gen0 = train_img_generator.flow_from_directory(batch_size = batch_size , directory = traindir , shuffle = True , target_size = (img_height , img_width) , class_mode = 'categorical')
#     valid_data_gen0 = valid_img_generator.flow_from_directory(batch_size = batch_size , directory = validdir , shuffle = False , target_size = (img_height , img_width) , class_mode = 'categorical')
#     test_data_gen0 = test_img_generator.flow_from_directory(batch_size = batch_size , directory = testdir , shuffle = False , target_size = (img_height , img_width) , class_mode = 'categorical')
    
#     for img_batch , label_batch in train_data_gen0:
#         print(f'Image_Batch_Size : {img_batch.shape}')
#         print(f'Label_Batch_Size : {label_batch.shape}')
#         break
#     T3 = time.time()

# print(f'Time for loading data on CPU:\n{T1-T0} seconds')

#%% Model Creation on GPU
# This 'valid' means no zero padding to data output.
with tf.device('/gpu:0'):
    train_data_gen = train_data_gen0
    valid_data_gen = valid_data_gen0
    test_data_gen = test_data_gen0
    
    model = Sequential([
        Conv2D(24, (11,11) , padding='same', activation='relu', 
                input_shape=(img_height, img_width ,3)),
       
        MaxPooling2D(pool_size = 2),
        
        Conv2D(32, (6,6) , padding='same', activation='relu'),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.BatchNormalization(),        
        MaxPooling2D(pool_size = 2 ),
        
        Conv2D(40, (5,5) , padding='same', activation='relu' ),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.BatchNormalization(),         
        MaxPooling2D(pool_size = 2),
        Flatten(),
        Dense(512, activation='relu'),  #kernel_regularizer=tf.keras.regularizers.l1(0.01)),
        # tf.keras.layers.Dropout(0.3),
        # tf.keras.layers.BatchNormalization(),
        Dense(128, activation='relu'),  #kernel_regularizer=tf.keras.regularizers.l1(0.01)),
        Dense(124, activation='relu'),  #kernel_regularizer=tf.keras.regularizers.l1(0.01))
    ])
    # kernel_regularizer=tf.keras.regularizers.l1(0.01)
    # activity_regularizer=tf.keras.regularizers.l2(0.01)
    
    model.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    model.summary()
#%% Editing Model
# with tf.device('/gpu:0'):
    ##### Didn't Work
    # Removing the dropout layer and adding batchNormalization layer
    # sparta = model
    # weighters = model.get_weights()
    # drop2batch = tf.keras.layers.BatchNormalization()(sparta.layers[-4].output)
    # sparta = tf.keras.Model(inputs = sparta.input , outputs=drop2batch)
    # # Adding other layers at the end
    # # dense1er = model.get_layer('dense_1')
    # # dense1 = dense1er.output
    # dense1 = Dense(128, activation='relu')(sparta.layers[-1].output)
    # sparta = tf.keras.Model(inputs = sparta.input , outputs = dense1)
    # dense2 = Dense(124, activation='relu')(sparta.layers[-1].output)
    # sparta = tf.keras.Model(inputs = sparta.input , outputs = dense2)
    # sparta.summary()
    # sparta.set_weights(weighters)    
    # path = Path(datadir)
    # path = path / 'ModelWeights' / 'Third Model Weights'
    # os.chdir(path)
    # sparta.load_weights('Third Model')
    
    # sparta.compile(optimizer=tf.keras.optimizers.SGD(),
    #       loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    #       metrics=['accuracy'])
#%% Training on GPU
with tf.device('/gpu:0'):
    train_data_gen = train_data_gen0
    valid_data_gen = valid_data_gen0
    test_data_gen = test_data_gen0
    T4 = time.time()
    epochs = 4
    history = model.fit_generator(train_data_gen , epochs = epochs , validation_data = valid_data_gen )#steps_per_epoch = 3500,validation_steps = 1500
    T5 = time.time()
print(f'Elapsed time for training data for {epochs} epochs:\n {T5-T4} seconds')

#%% Model Saving

path = Path(datadir)

os.chdir(path)

if 'ModelStructure' not in os.listdir():
    os.mkdir('ModelStructure')

path = path / 'ModelStructure'

os.chdir(path)

model.save('Final Model')

path = Path(datadir)

os.chdir(path)

if 'ModelWeights' not in os.listdir():
    os.mkdir('ModelWeights')

os.mkdir('Third Model Weights')

path = path / 'ModelWeights' / 'Third Model Weights'

os.chdir(path)

model.save_weights('Third Model')

path = Path(datadir)

os.chdir(path)

if 'Histories' not in os.listdir():
    os.mkdir('Histories')

path = path / 'Histories'

os.chdir(path)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

# acc = acc1
# val_acc = val_acc1
# loss = loss1
# val_loss = val_loss1

# acc.extend(acc0)
# val_acc.extend(val_acc0)
# loss.extend(loss0)
# val_loss.extend(val_loss0)

f = open('acc2' , 'wb')
pickle.dump(acc,f)
f.close()

f = open('val_acc2' , 'wb')
pickle.dump(val_acc,f)
f.close()

f = open('loss2' , 'wb')
pickle.dump(loss,f)
f.close()

f = open('val_loss2' , 'wb')
pickle.dump(val_loss,f)
f.close()

#%% Model Loading

path = Path(datadir)

path = path / 'ModelStructure'

os.chdir(path)

Loaded_Model = tf.keras.models.load_model('Third Model')

model = Loaded_Model


path = Path(datadir)

path = path / 'Histories'

os.chdir(path)

f = open('acc2' , 'rb')
acc2 = pickle.load(f)
f.close()

f = open('val_acc2' , 'rb')
val_acc2 = pickle.load(f)
f.close()

f = open('loss2' , 'rb')
loss2 = pickle.load(f)
f.close()

f = open('val_loss2' , 'rb')
val_loss2 = pickle.load(f)
f.close()

#%% Validation
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

# acc = acc1
# val_acc = val_acc1
# loss = loss1
# val_loss = val_loss1

# acc.extend(acc0)
# val_acc.extend(val_acc0)
# loss.extend(loss0)
# val_loss.extend(val_loss0)

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()



