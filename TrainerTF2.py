import tensorflow.keras as keras
import os
from UTILS_TF2 import dat_loader, tcn_model_builder

# GPU = 0
# os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU)
# print(f'GPU: {GPU}')

#################################################################################################################
#  This python script will train the TCN Model. The basic TCN code was borrowed from Philippe Rémy's github:    #
#  https://github.com/philipperemy/keras-tcn/blob/master/tcn/tcn.py                                             #
#################################################################################################################

# Define Folders for saving Models/Logs
model_folder = 'models'
log_folder = 'logs'
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

# Define Training Set
dat_name = '360sec_200buf_0.02fmin_300aw_0.02dec_10amp_max'
dat_dir = os.path.join('data', dat_name)
# stations = ['PDAR','MKAR','TXAR','ABKAR','ASAR','ILAR']
stations = ['MKAR']

x_train, y_train, x_test, y_test = dat_loader(stations, dat_dir)

# Define the TCN Hyper-parameters
d = [1, 2, 4, 8]    # The list of the dilations. Example is: [1, 2, 4, 8].
k = 16              # The number of samples in each kernel
s = 12              # The number of stacks of Residual Blocks in the TCN
f = 20              # The number of filters in each Layer of the TCN

# Compile the TCN Model
model, param_str = tcn_model_builder(d, k, s, f, x_train.shape[1])

# Define the Keras Callbacks for saving, logging and stopping the TCN model during training
param_str = str(stations).replace('[', '').replace(']', '').replace(',', '+').replace(' ', '').replace("'", '') + '_' + param_str
name = '--'.join([dat_name, param_str])
model_filename = os.path.join(os.getcwd(), model_folder, 'best_model_' + name + '.h5')
tensor_filename = os.path.join(os.getcwd(), log_folder, name)
sv = keras.callbacks.ModelCheckpoint(filepath=model_filename, save_best_only=True, save_weights_only=True)
tbd = keras.callbacks.TensorBoard(log_dir=tensor_filename)
stp = keras.callbacks.EarlyStopping(patience=10)

# Train the TCN Model
model.summary()
print('training model {}'.format(model_filename))
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, callbacks=[tbd, sv, stp], batch_size=20)



