# -*- coding: utf-8 -*-

import sys
import os
from datetime import datetime
from pathlib import Path

import numpy as np

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

from M3SPADA_model import M3SPADAModel


# Computing on CPU or GPU
if sys.argv[9] == "-1":
    # Computation on CPU only, no use of GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    # Computation on GPU with index sys.argv[9]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[9]
    # Avoid use of full GPU RAM if not necessary
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)


# Global variables
loss_ts_set = []
loss_vhsr_set = []
loss_fus_set = []
loss_pred_set = []
loss_domain_set = []
loss_pred_set_PL = []
loss_combined_set = []
target_f1 = []
PL_accuracy = []


def getBatch(X, i, batch_size):
    start_id = (i*batch_size)
    end_id = min((i+1) * batch_size, X.shape[0])
    batch_x = X[start_id:end_id]
    return batch_x


def train_M3SPADA(model, optimizer, s_X, s_VHSR_X, s_y, t_X, t_VHSR_X,
                      t_y, num_epochs, batch_size, bn_flag, loss_fn):
    """"Training function for M3SPADA model"""

    global loss_pred_set
    global loss_pred_set_PL, loss_combined_set, target_f1, PL_accuracy
    global alpha1, alpha2
    
    epochs = range(num_epochs)
    
    nb_samples = s_X.shape[0]
    iterations = nb_samples / batch_size
        
    if nb_samples % batch_size != 0:
        iterations += 1
    
    for epoch in epochs:
        test_X = t_X.copy()
        test_VHSR_X = t_VHSR_X.copy()
        test_y = t_y.copy()
        
        s_X, s_VHSR_X, s_y, t_X, t_VHSR_X, t_y \
            = shuffle(s_X, s_VHSR_X, s_y, t_X, t_VHSR_X, t_y)

        alpha = (float(epoch) / num_epochs)

        # lamb_da (lambda_da) is for a progressive use of gradient reversal
        # Cf. DANN paper
        lamb_da = 2 / (1 + np.exp(-10 * (float(epoch) / num_epochs),
                                  dtype=np.float32)) - 1
        lamb_da = lamb_da.astype('float32')
        
        PL_global_set = []
        test_y_2cpe = []
                                
        for ibatch in range(int(iterations)):
            batch_source_ts = getBatch(s_X, ibatch, batch_size)
            batch_source_VHSR = getBatch(s_VHSR_X, ibatch, batch_size)
            batch_source_y = getBatch(s_y, ibatch, batch_size)

            batch_target_ts = getBatch(t_X, ibatch, batch_size)
            batch_target_VHSR = getBatch(t_VHSR_X, ibatch, batch_size)
            batch_target_y = getBatch(t_y, ibatch, batch_size)
                        
            with tf.GradientTape() as tape:
                # Model application for the source domain
                ts_class_source, vhsr_class_source, lpred_source, dpred_source\
                    = model([batch_source_ts, batch_source_VHSR],
                            bn_flag, lamb_da)

                # # Model application for the target domain    
                ts_class_target, vhsr_class_target, lpred_target, dpred_target\
                    = model([batch_target_ts, batch_target_VHSR],
                            bn_flag, lamb_da)
                
                # Pseudo label selection on target domain
                lpred_src = np.argmax(lpred_source, axis=1)
                lpred_tgt = np.argmax(lpred_target, axis=1)
                first_cond = (lpred_src == batch_source_y)
                second_cond = (lpred_src == lpred_tgt)
                result = (first_cond & second_cond).astype(int)
                
                # Calculation of loss on label prediction on source domain
                loss_src_ts = loss_fn(batch_source_y, ts_class_source)
                loss_src_vhsr = loss_fn(batch_source_y, vhsr_class_source)
                loss_src_fus = loss_fn(batch_source_y, lpred_source)                       
                loss_pred = alpha1 * loss_src_ts\
                    + alpha2 * loss_src_vhsr + loss_src_fus

                # Calculation of loss on domain prediction
                loss_domain\
                    = loss_fn(tf.concat([np.ones(batch_source_ts.shape[0]), 
                                         np.zeros(batch_target_ts.shape[0])], 
                                        axis=0),
                              tf.concat([dpred_source, dpred_target], 
                                        axis=0))
                
                # Calculation of loss on pseudo-labels on source domain
                lpred_tgt_ts = np.argmax(ts_class_target, axis=1)
                lpred_tgt_vhsr = np.argmax(vhsr_class_target, axis=1)
                loss_pred_PL\
                    = alpha1 * loss_fn(lpred_tgt_ts,
                                       ts_class_target, sample_weight=result)\
                        + alpha2 * loss_fn(lpred_tgt_vhsr, vhsr_class_target,
                                         sample_weight=result)\
                            + loss_fn(lpred_tgt, lpred_target,
                                      sample_weight=result)
                # Calculation of combined loss
                loss_combined = (1 - alpha) * (loss_pred + loss_domain)\
                    + alpha * loss_pred_PL
                
                # Pseudo-labels and true labels on target domain aggregation
                PL_global_set.append(lpred_tgt[np.where(result == 1)])
                test_y_2cpe.append(batch_target_y[np.where(result == 1)])
                  
            grads = tape.gradient(loss_combined, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            ts_loss(loss_src_ts)
            vhsr_loss(loss_src_vhsr)
            fus_loss(loss_src_fus)
            train_loss(loss_pred)
            domain_loss(loss_domain)
            train_loss_PL(loss_pred_PL)
            value_loss(loss_combined)

        # Evaluation on full target domain
        _, _, pred_test_target, _ = model.predict([test_X, test_VHSR_X],
                                                  batch_size=1024)
        fscoreT = f1_score(test_y, np.argmax(pred_test_target, axis=1),
                           average="weighted")

        # Computation of global accuracy on target domain PL
        PL_global_set\
            = np.concatenate(np.asarray(PL_global_set, dtype=object), axis=0)
        PL_counter = PL_global_set.shape[0]    
        test_y_2cpe\
            = np.concatenate(np.asarray(test_y_2cpe, dtype=object), axis=0)
        accuracy_PL = accuracy_score(test_y_2cpe, PL_global_set)

        # Printing of different scores
        print("Epoch %d | TS LOSS %.5f | VHSR LOSS %.5f | FUS LOSS %.5f |"
              " TRAIN LOSS %.5f | DOMAIN LOSS %.5f |"
              " PL_LOSS %.5f | TRAIN+DOMAIN+PL LOSS %.5f "
              "| TARGET F1-score %.3f | #PL %d | PL ACC %.3f"
              % (epoch, ts_loss.result(), vhsr_loss.result(),
                 fus_loss.result(), train_loss.result(), domain_loss.result(),
                 train_loss_PL.result(), value_loss.result(),
                 fscoreT, PL_counter, accuracy_PL))
        PL_accuracy.append(accuracy_PL)
      
        # Scores saving to use for graphic
        loss_ts_set.append(ts_loss.result())
        loss_vhsr_set.append(vhsr_loss.result())
        loss_fus_set.append(fus_loss.result())
        loss_pred_set.append(train_loss.result())
        loss_domain_set.append(domain_loss.result())
        loss_pred_set_PL.append(train_loss_PL.result())
        loss_combined_set.append(value_loss.result())
        target_f1.append(fscoreT)
                    
        
# ################################
# Script main body

# Testing of script arguments number
if len(sys.argv) != 10:
    print("!!! Error : wrong number of arguments !!!")
    print("!!! Stop script                       !!!")
    exit()

# Retrieval of values passed through script arguments
s_year = sys.argv[1]  # Year of source domain
t_year = sys.argv[2]  # Year of target domain
learning_rate = float(sys.argv[3])
num_epochs = int(sys.argv[4])
batch_size = int(sys.argv[5])
alpha1 = float(sys.argv[6])  # Weight for TS classifier
alpha2 = float(sys.argv[7])  # Weight for VHR/VHSR classifier
patch_size = int(sys.argv[8])
s_path = s_year
t_path = t_year

a1 = 25  # Size of initial patch
b = int((a1 - patch_size) / 2)
c = b + patch_size

start_date = datetime.now()
print("Calculation completed at: %s" % str(start_date))

# Path to save the model
pth2MD = "Model_Data/"
Path(pth2MD).mkdir(exist_ok=True, parents=True)

# Path to save results
pth2R = "Results/"
Path(pth2R).mkdir(exist_ok=True, parents=True)

# Useful for retrieving identify and retrieve results
suffix = '_'+sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3]+'_'+sys.argv[4]+'_'\
        +sys.argv[5]+'_'+sys.argv[6]+'_'+sys.argv[7]\
            +'_'+sys.argv[8]

# Source and target data
# Let N the number of pixels (samples)
# Let T the time serie for a pixel
# Let s_X a ndarray of shape (N, T) containing source domain observation data for time series
# Let s_VHSR_X_tmp a ndarray of shape (N, 25, 25) containing source domain observation data for VHR/VHSR
# Each sample of s_VHSR_X_tmp is a patch of 25 x 25 of VHR images
# Let t_X a ndarray of shape (N, T) containing target domain observation data
# Let t_VHSR_X_tmp a ndarray of shape (N, 25, 25) containing target domain observation data for VHR/VHSR
# Each sample of t_VHSR_X_tmp is a patch of 25 x 25 of VHR images
# 'float32' is dtype for s_X, s_VHSR_X_tmp and t_Xn t_VHSR_X_tmp
# Let s_y a ndarray of shape (N,) containing source domain true labels
# Let t_y a ndarray of shape (N,) containing target domain true labels
# To reduce the need of RAM (CPU and GPU) we define dtype of s_y and t_y as
# 'uint8'. It's possible because le number of classes <= 256
# If class index starts from 1 and not 0, it's necessary to decrease all class
# indexes by one

# Resizing 25 x 25 VHR patches to patch_size x patch_size VHR patches
# patch_size = 15 is the best setting for our datasets
# Be careful, patch encoder in M3SPADA model has been designed to deal with this setting
s_VHSR_X = s_VHSR_X_tmp[:, b:c, b:c, :]
t_VHSR_X = t_VHSR_X_tmp[:, b:c, b:c, :]

nb_class = len(np.unique(s_y))

model = M3SPADAModel(nb_class, nb_units=256, drop_val=0.5)
model_file_name = "M3SPADA"
sd = model_file_name + suffix + "/"

# Use to distinguish training and prediction mode for Batch normalization
# and Dropout layers
bn_flag = True

# Choice of loss function
loss_fn = keras.losses.SparseCategoricalCrossentropy()

# Choice of optimizer
optimizer = keras.optimizers.Adam(learning_rate)

# Choice of metrics
ts_loss = tf.keras.metrics.Mean(name='ts_loss')
vhsr_loss = tf.keras.metrics.Mean(name='vhsr_loss')
fus_loss = tf.keras.metrics.Mean(name='fusion_loss')
train_loss = tf.keras.metrics.Mean(name='train_loss')
domain_loss = tf.keras.metrics.Mean(name='domain_loss')
train_loss_PL = tf.keras.metrics.Mean(name='train_loss_PL')
value_loss = tf.keras.metrics.Mean(name='value_loss')

# Training phase
print("###########################")
print("Start of the training loop")
print("---------------------------")
print("Model: ", model_file_name)
print("without regularization")
print("Source data path :", s_path)
print("Target data path:", t_path)
print("Source domain: ", s_year)
print("Target domain: ", t_year)
print("Learning rate: ", learning_rate)
print("Number of epochs: ", num_epochs)
print("Batch size: ", batch_size)
print("(alpha1, alpha2) = ", alpha1, alpha2)
print("Patch x*x with x:", patch_size)

train_M3SPADA(model, optimizer, s_X, s_VHSR_X, s_y, t_X, t_VHSR_X,
                  t_y, num_epochs, batch_size, bn_flag, loss_fn)

# Model backup
model.save_weights(pth2MD+sd+model_file_name)

# Graphic regarding training phase
x_axis = [j for j in range(0, num_epochs)]
plt.plot(x_axis, loss_ts_set, label="loss_ts")
plt.plot(x_axis, loss_vhsr_set, label="loss_vhsr")
plt.plot(x_axis, loss_fus_set, label="loss_fus")
plt.plot(x_axis, loss_pred_set, label="loss_train")
plt.plot(x_axis, loss_domain_set, label="loss_domain")
plt.plot(x_axis, loss_pred_set_PL, label="loss_train_PL")
plt.plot(x_axis, loss_combined_set, label="loss_combined")
plt.plot(x_axis, target_f1, label="f1_target")
plt.plot(x_axis, PL_accuracy, label="accuracy_PL")
plt.legend()
plt.savefig(pth2R + model_file_name + suffix + '.png')
plt.close()

print("End of the training loop")
print("#########################")

# Evaluation phase
print("#############################################")
print("Start of the evaluation on full target domain")
print("---------------------------------------------")
_, _, pred_test_target, _ = model.predict([t_X, t_VHSR_X], batch_size=1024)
       
# Backup for further analysis of t_X t_VHSR_X, t_y and label prediction on t_X
Path(pth2R+sd).mkdir(exist_ok=True, parents=True)
np.save(pth2R+sd+"best_"+model_file_name+"-t_X", t_X)
np.save(pth2R+sd+"best_"+model_file_name+"-t_VHSR_X", t_VHSR_X)
np.save(pth2R+sd+"best_"+model_file_name+"-t_y", t_y)
np.save(pth2R+sd+"best_"+model_file_name+"-predictions", pred_test_target)

# Calculation of scores
y_pred = np.argmax(pred_test_target, axis=1)
accuracy_TD = np.round(accuracy_score(t_y, y_pred), 3)
f1_score_TD = np.round(f1_score(t_y, y_pred, average='weighted'), 3)
kappa_TD = np.round(cohen_kappa_score(t_y, y_pred), 3)

print('\n')
print('**********************************************************************')
print("Final results : F1 score (average='weighted') =%1.3f" % f1_score_TD)
print('**********************************************************************')
print("Final results : Accuracy =%1.3f" % accuracy_TD)
print('**********************************************************************')
print("Final results : Cohen's Kappa score =%1.3f" % kappa_TD)
print('**********************************************************************')
print('\n')

print("End of the evaluation on full target domain")
print("#############################################")

end_date = datetime.now()
print("Calculation completed at: %s" % str(end_date))
print("Calculation time: %s" % str(end_date - start_date))
