import os
import numpy as np
import pandas as pd
from UTILS import target_exp, my_generator, gen_single_obs, drop_csv
import obspy
import datetime

#################################################################################################################
#  This python script will preprocess the raw waveforms and arrival catalogs into suitable Train and Test sets. #
#  The Training set consists of fixed length X,Y pairs of sequence data.                                        #
#  The Test set consists of a continuous stream of sequence data.                                               #
#################################################################################################################

# Define Parameters of Training Datasets
train_stations = ['MKAR']
st_yyyy, st_mm, en_yyyy, en_mm = [2010, 1, 2015, 1]
train_st = datetime.datetime(st_yyyy, st_mm, 1)
train_en = datetime.datetime(en_yyyy, en_mm, 1)

# Define Parameters of Testing Datasets
test_stations = ['PDAR']
st_yyyy, st_mm, en_yyyy, en_mm = [2015, 1, 2015, 2]
test_st = datetime.datetime(st_yyyy, st_mm, 1)
test_en = datetime.datetime(en_yyyy, en_mm, 1)
test_name = '_{0}_{1}_{2}_{3}'.format(st_yyyy, st_mm, en_yyyy, en_mm)

# Define Parameters of Training waveforms
fs=40               # Sample Rate of Training waveforms
win_time = 360      # Seconds in Training waveforms
buffer = 200        # Seconds till arrival in Training waveforms
freq = .02          # Lowpass filter cutoff frequency for Training waveforms
win_samps = int(win_time * fs)

# Define Parameters of Target Exponential
aw = 300            # Samples in target exponential
amp = 10            # Amplitude of target exponential
decay = .02         # Decay Rate of target exponential
alert = target_exp(aw, amp, decay)

# Define Directory to Store Dataset
data_dir = 'data/{0}sec_{1}buf_{2}fmin_{3}aw_{4}dec_{5}amp_max'.format(win_time, buffer, freq, aw, decay, amp)


# Generate Training Set
for station_name in train_stations:
    if not os.path.isfile(os.path.join(data_dir, 'x_val_{}.npy'.format(station_name))):
        print('generating training data for', station_name)
        cat_dir = 'data/Cat_' + station_name
        file = '{}ArrivalsSNR.csv'.format(station_name)
        filename = os.path.join(cat_dir, file)
        drop_csv(filename, win_time - buffer - 2*aw/fs)

        cat = pd.read_csv(filename, index_col=0)
        cat['TIME'] = pd.to_datetime(cat['TIME'])
        cat['EVENTID'] = cat['EVENTID'].astype('int')
        cat.fillna('', inplace=True)
        cat = cat.loc[(cat.TIME > train_st) & (cat.TIME < train_en)]
        trunc_cat = cat.loc[cat.DROP == 0]

        bs = 2000
        gen = my_generator(full_cat=cat, trunc_cat=trunc_cat, batch_size=bs, win_time=win_time, alert_func=alert, buffer=buffer, f_min=freq)


        print('generating val - batch: 01')
        X_val, Y_val = next(gen)

        print('generating trn - batch: 02')
        X_train, Y_train = next(gen)

        for i in range(len(trunc_cat) // bs - 2):
            print('generating trn - batch: {:02}'.format(i + 3))
            new_X, new_Y = next(gen)
            X_train = np.concatenate((X_train, new_X))
            Y_train = np.concatenate((Y_train, new_Y))

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        np.save(os.path.join(data_dir, 'x_val_{}.npy'.format(station_name)), X_val)
        np.save(os.path.join(data_dir, 'y_val_{}.npy'.format(station_name)), Y_val)
        np.save(os.path.join(data_dir, 'x_train_{}.npy'.format(station_name)), X_train)
        np.save(os.path.join(data_dir, 'y_train_{}.npy'.format(station_name)), Y_train)
        X_val, Y_val, X_train, Y_train, new_X, new_Y = [None] * 6


# Generate Test Set
for station_name in test_stations:
    if not os.path.isfile(os.path.join(data_dir, 'x_test_' + station_name + test_name + '.npy')):
        print('generating test data for', station_name)
        cat_dir = 'data/Cat_' + station_name
        file = '{}ArrivalsSNR.csv'.format(station_name)
        filename = os.path.join(cat_dir, file)

        cat = pd.read_csv(filename, index_col=0)
        cat['TIME'] = pd.to_datetime(cat['TIME'])
        cat['EVENTID'] = cat['EVENTID'].astype('int')
        cat = cat.loc[(cat.TIME > test_st) & (cat.TIME < test_en)]

        stream = obspy.core.stream.Stream
        for st, mseed_file in enumerate(sorted(set([name.split(' ')[0] for name in cat.FILE.values.tolist()]))):
            if st == 0:
                stream = obspy.core.read(mseed_file)
            else:
                stream = stream + obspy.core.read(mseed_file)

        stream.merge()
        tr = stream[0]
        win_time = tr.stats['endtime'] - tr.stats['starttime']

        X_test_full, Y_test_full, test_cat = gen_single_obs(tr=tr, cat=cat, win_time=win_time, alert_func=alert,
                                                            f_min=freq, test_set=True)

        test_cat.to_csv(os.path.join(data_dir, 'test_cat_' + station_name + test_name + '.csv'))
        np.save(os.path.join(data_dir, 'x_test_' + station_name + test_name + '.npy'), X_test_full.reshape(1, -1, 1))
        np.save(os.path.join(data_dir, 'y_test_' + station_name + test_name + '.npy'), Y_test_full.reshape(1, -1, 1))
        X_test_full, Y_test_full = [None] * 2
        print('test')

