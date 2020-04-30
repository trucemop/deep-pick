from scipy import signal as sig
import urllib.request
import pandas as pd
import re
import os
import errno
import obspy
import numpy as np
from pathlib import Path
import sys
import datetime
from philipperemy_keras_tcn import dilated_tcn
import matplotlib.pyplot as plt
import matplotlib
from phasepicker import fbpicker, ktpicker, aicdpicker
from obspy.core import *
import time
import multiprocessing
import random
from multiprocessing import Queue


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred


def url_maker(st, yr, mm, pc, pt):
    if (mm == 12) & (pc == pt - 1):
        en_yr = yr + 1
        en_mm = 1
    elif (mm != 12) & (pc == pt - 1):
        en_yr = yr
        en_mm = mm + 1
    else:
        en_mm = mm
        en_yr = yr

    dd = np.ones(pt + 1)
    for i in range(1, pt):
        dd[i] = 30 * i / pt
    dd = dd.astype('int')
    url = 'http://www.isc.ac.uk/cgi-bin/web-db-v4?iscreview=on&out_format=CSV&ttime=on&ttres=on&tdef=on&phaselist=&stnsearch=STN&sta_list=' + st + '&stn_ctr_lat=&stn_ctr_lon=&stn_radius=&max_stn_dist_units=deg&stn_top_lat=&stn_bot_lat=&stn_left_lon=&stn_right_lon=&stn_srn=&stn_grn=&bot_lat=&top_lat=&left_lon=&right_lon=&ctr_lat=&ctr_lon=&radius=&max_dist_units=deg&searchshape=GLOBAL&srn=&grn=&start_year=' + str(
        yr) + '&start_month=' + str(mm) + '&start_day=' + str(dd[pc]) + '&start_time=00%3A00%3A00&end_year=' + str(
        en_yr) + '&end_month=' + str(
        en_mm) + '&end_day=' + str(dd[
                                       pc + 1]) + '&end_time=00%3A00%3A00&min_dep=&max_dep=&min_mag=&max_mag=&req_mag_type=Any&req_mag_agcy=Any&include_links=on&request=STNARRIVALS'
    return url


def url_open(url, filename):
    # Make 10 attempts at downloading the catalog
    reattempt = 0
    while reattempt < 10:

        # Try 10 times to retrieve the url
        retrycount = 0
        s = None
        while s is None:
            try:
                s = urllib.request.urlretrieve(url, filename)
            except Exception as e:
                print(str(e))
                retrycount += 1
                if retrycount > 5:
                    print(" download failed")
                    # silentremove(filename)

        # Check to see if the retrieval was rejected by the server
        # If the retrieval was rejected, reattempt!
        if open(filename).readlines()[23][:5] == 'Sorry':
            reattempt += 1
            print(' reattempt', reattempt, '...', sep='', end='')
            sys.stdout.flush()
            time.sleep(60)
        else:
            return
    print(" download failed")
    # silentremove(filename)


def clean_csv(filename):
    lines = open(filename).readlines()
    first_chars = [line[:6] for line in lines]

    idx_st = [i for i in range(len(first_chars)) if first_chars[i] == 'EVENTI']
    idx_en = [i for i in range(len(first_chars)) if first_chars[i] == 'STOP\n']

    if len(idx_en) > 0:
        lines = lines[idx_st[0]:idx_en[0] - 1]
        lines = [re.sub('\<(.*?)\>', '', line, count=0, flags=0) for line in lines]

        with open('tmp.csv', 'w') as f:
            f.writelines(lines)

        cat_event = pd.read_csv('tmp.csv')

        cat_event.columns = cat_event.columns.str.replace('.', '_')
        cat_event.columns = cat_event.columns.str.replace(' ', '')
        cat_event = cat_event.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        cat_event['TIME'] = (cat_event['DATE'] + ' ' + cat_event['TIME'])
        cat_event['TIME'] = pd.to_datetime(cat_event['TIME'])
        for col in ['LAT', 'LON', 'ELEV', 'AMPLITUDE', 'PER', 'MAG']:
            cat_event[col] = pd.to_numeric(cat_event[col])

        return cat_event


def monthly_catalog_downloader(yr, mm, station_name, filename, pt=5):
    print('downloading ', filename, '...', sep='', end='')
    sys.stdout.flush()
    if Path(filename).is_file():
        print(" already downloaded")
    else:

        cat = pd.DataFrame()
        for i in range(pt):
            url = url_maker(station_name, yr, mm, i, pt)
            url_open(url, 'tmp.csv')
            cat = cat.append(clean_csv('tmp.csv'))
        cat.to_csv(filename)
        print(" download complete")


def SNR_calc(tr, arr_time):
    fs = tr.stats['sampling_rate']

    # Calculate SNR for Teleseismic Signals
    p_win = 5
    n_win = 40
    snr_samp = int(n_win * fs)
    tr_slice = tr.slice(starttime=arr_time - n_win, endtime=arr_time + p_win)
    tr_slice = tr_slice.detrend('constant')
    tr_slice = tr_slice.normalize()

    tr_filt = tr_slice.filter('bandpass', freqmin=1, freqmax=4)
    sig_P = np.abs(np.dot(tr_filt[snr_samp:], tr_filt[snr_samp:]) / (p_win * fs))
    nos_P = np.abs(np.dot(tr_filt[:snr_samp], tr_filt[:snr_samp]) / (n_win * fs))
    SNR0 = 10 * np.log10(sig_P / nos_P)

    tr_filt = tr_slice.filter('bandpass', freqmin=1.8, freqmax=4.2)
    sig_P = np.abs(np.dot(tr_filt[snr_samp:], tr_filt[snr_samp:]) / (p_win * fs))
    nos_P = np.abs(np.dot(tr_filt[:snr_samp], tr_filt[:snr_samp]) / (n_win * fs))
    SNR1 = 10 * np.log10(sig_P / nos_P)
    SNR = max(SNR0, SNR1)
    SNR = SNR1
    return SNR


def SNR_catalogger(miniseed_filepath, catalog):
    big_cat = pd.DataFrame()
    miniseed_stream = obspy.core.read(miniseed_filepath)
    for i, tr in enumerate(miniseed_stream):

        tr_st = tr.stats['starttime'] + 2000
        tr_en = tr.stats['endtime'] - 2000
        fs = tr.stats['sampling_rate']

        cat = catalog.loc[(catalog['TIME'] > tr_st.datetime) & (catalog['TIME'] < tr_en.datetime)].reset_index(
            drop=True)
        print(miniseed_filepath + ': {0:.0f}'.format(len(cat)))
        SNR = np.zeros((len(cat)))
        FILE = []

        for index, arrival in cat.iterrows():
            arr_time = obspy.UTCDateTime(arrival.TIME)
            SNR[index] = SNR_calc(tr, arr_time)
            FILE.append(miniseed_filepath + ' ' + str(i))

        cat = cat.assign(FILE=pd.Series(FILE).values)
        cat = cat.assign(SNR=pd.Series(SNR).values)

        big_cat = big_cat.append(cat)
    return big_cat


def monthly_miniseed_downloader(c, year, month, filename, station, network):
    print('downloading ', filename, '...', sep='', end='')
    sys.stdout.flush()
    st_time = obspy.UTCDateTime(str(year) + "{:02}".format(month) + "-01T00:00:00.0")
    if month == 12:
        en_time = obspy.UTCDateTime(str(year + 1) + "01-01T00:00:00.0")
    else:
        en_time = obspy.UTCDateTime(str(year) + "{:02}".format(month + 1) + "-01T00:00:00.0")

    sta = station
    cha = "BHZ"
    net = network
    loc = ""

    if os.path.isfile(filename):
        print(" already downloaded")
    else:
        try:
            event_window = c.get_waveforms(network=net, location=loc,
                                           station=sta, channel=cha,
                                           starttime=st_time, endtime=en_time)

            event_window.write(filename, format="MSEED")
            print(" download complete")
        except:
            print(" download failed")



def merge_csv(cat_dir, FULL_CAT_filename, discriminant='_arrivals.csv'):
    my_files = sorted(os.listdir(cat_dir))
    my_files = [file for file in my_files if (file[-len(discriminant):] == discriminant)]

    CAT = pd.DataFrame()

    for file in my_files:
        print('Appending {}...'.format(file))
        filename = os.path.join(cat_dir, file)
        cat = pd.read_csv(filename, index_col=0)
        cat['TIME'] = pd.to_datetime(cat['TIME'])
        cat['EVENTID'] = cat['EVENTID'].astype('int')

        CAT = CAT.append(cat)
        CAT.reset_index(drop=True, inplace=True)

    print('saving {}...'.format(FULL_CAT_filename))
    CAT.to_csv(FULL_CAT_filename)

def snr_csv(snr_cat_filename, full_cat_filename, dat_dir):

    cat = pd.read_csv(full_cat_filename, index_col=0)
    cat['TIME'] = pd.to_datetime(cat['TIME'])
    cat['EVENTID'] = cat['EVENTID'].astype('int')

    snr_cat = pd.DataFrame()

    for file in sorted(os.listdir(dat_dir)):
        filename = os.path.join(dat_dir, file)
        snr_cat = snr_cat.append(SNR_catalogger(filename, cat))

    snr_cat.to_csv(snr_cat_filename)


def detect_csv(cat_filename, arrayname, sitename):
    cat = pd.read_csv(cat_filename, index_col=0)
    cat['TIME'] = pd.to_datetime(cat['TIME'])
    cat['EVENTID'] = cat['EVENTID'].astype('int')

    arr_cat = cat.loc[(cat.STA == arrayname)]
    arr_cat.reset_index(drop=True, inplace=True)

    site_cat = cat.loc[cat.STA == sitename]
    site_cat.reset_index(drop=True, inplace=True)

    delta = datetime.timedelta(seconds=1)

    DETECT = np.zeros((len(arr_cat)))

    for index, arr_event in arr_cat.iterrows():
        match_cat = site_cat.loc[
            (site_cat['TIME'] > arr_event['TIME'] - delta) & (site_cat['TIME'] < arr_event['TIME'] + delta)]
        if len(match_cat) > 0:
            my_delta = datetime.timedelta.total_seconds(match_cat.iloc[0]['TIME'] - arr_event['TIME'])
            DETECT[index] = 1

    arr_cat = arr_cat.assign(DETECT=pd.Series(DETECT).values)
    arr_cat.to_csv(cat_filename.split('.')[0] + '_DET.' + cat_filename.split('.')[1])



def gen_single_obs(tr, cat, win_time, alert_func, f_min, test_set=False):
    win_st = tr.stats['starttime']
    win_en = tr.stats['endtime'] - 1
    fs = tr.stats['sampling_rate']

    win_samps = int(win_time * fs)
    Y = np.zeros((win_samps, 1))
    stream = tr.split()
    for trace in stream:
        trace.filter("bandpass", freqmin=f_min, freqmax=10)
    stream.merge()
    tr = stream[0]
    X = tr.data[:win_samps]
    if np.ma.is_masked(X):
        X = tr.data[:win_samps].filled(fill_value=0)

    X = X.reshape((win_samps, 1))
    win_cat = cat.loc[(cat['TIME'] > win_st.datetime) & (cat['TIME'] < win_en.datetime)]
    #win_cat['IDX'] = ''

    for index, arrival in win_cat.iterrows():
        arr_time = obspy.UTCDateTime(arrival.TIME)
        arr_idx = int(win_samps * (arr_time - win_st) / win_time)
        end_idx = min(len(alert_func), win_samps - arr_idx)
        Y[arr_idx:arr_idx + end_idx, 0] = np.maximum(Y[arr_idx:arr_idx + end_idx, 0], alert_func[:end_idx])
        win_cat.loc[index, 'IDX'] = arr_idx

    if test_set:
        return X, Y, win_cat
    else:
        return X, Y


def my_generator(full_cat, trunc_cat, batch_size, win_time, alert_func, buffer=1, f_min=1, fs=40):
    i = 0

    while 1:

        # generate a batch of rows from the catalog
        rows = np.arange(i, int(min(i + batch_size, len(trunc_cat))))
        if len(rows) < batch_size:
            rows = np.hstack((np.arange(0, batch_size - len(rows)), rows)).astype('int32')

        i = int((i + batch_size) % len(trunc_cat))

        # declare the output variables for this batch
        win_samps = int(win_time * fs)
        X = np.zeros((batch_size, win_samps, 1))
        Y = np.zeros((batch_size, win_samps, 1))
        SNR = np.zeros(batch_size)
        EVENT = np.zeros(batch_size)
        FILE = ''

        # generate positive examples
        for j, row in enumerate(rows):
            event = trunc_cat.iloc[row]
            SNR[j] = event.SNR
            EVENT[j] = event.EVENTID
            if FILE != event.FILE:
                tr_file, tr_idx = event.FILE.split(' ')
                stream = obspy.core.read(tr_file)
                tr = stream[int(tr_idx)]
                FILE = event.FILE

            t = obspy.UTCDateTime(event.TIME) - buffer
            tr_slice = tr.slice(starttime=t, endtime=t + win_time + 1)
            X[j], Y[j] = gen_single_obs(tr_slice, full_cat, win_time, alert_func, f_min)

        yield X, Y

def target_exp(aw=10, amp=1, decay=1):
    x_right = amp*np.exp(-decay*np.arange(aw))
    x_left = np.flip(x_right,0)
    x = np.hstack((x_left,x_right))
    return x


def drop_csv(cat_filename, margin):
    cat = pd.read_csv(cat_filename, index_col=0)
    cat['TIME'] = pd.to_datetime(cat['TIME'])
    cat['EVENTID'] = cat['EVENTID'].astype('int')

    cat = cat.sort_values(by=['TIME'])
    cat.reset_index(drop=True, inplace=True)
    margin = datetime.timedelta(seconds=margin).total_seconds()
    my_time = cat.loc[0].TIME

    DROP = np.zeros(len(cat))
    for row in range(1, len(cat)):
        next_time = cat.loc[row].TIME
        delta = (next_time - my_time).total_seconds()
        if delta < margin:
            DROP[row] = 1
        else:
            my_time = cat.loc[row].TIME

    cat = cat.assign(DROP=pd.Series(DROP).values)
    print('Length of Catalog:', int(len(DROP) - np.sum(DROP)))
    cat.to_csv(cat_filename)


def drop_cat(cat, margin):
    cat = cat.sort_values(by=['TIME'])
    cat.reset_index(drop=True, inplace=True)
    margin = datetime.timedelta(seconds=margin).total_seconds()
    my_time = cat.loc[0].TIME

    DROP = np.zeros(len(cat))
    for row in range(1, len(cat)):
        next_time = cat.loc[row].TIME
        delta = (next_time - my_time).total_seconds()
        if delta < margin:
            DROP[row] = 1
        else:
            my_time = cat.loc[row].TIME

    cat = cat.assign(DROP=pd.Series(DROP).values)
    trunc_cat = cat.loc[cat.DROP == 0]
    return trunc_cat


def dat_loader(station_list, fld):

    size = np.load(os.path.join(fld, 'x_val_{}.npy'.format(station_list[0]))).shape[1]
    x_train = np.empty((1, size, 1))
    y_train = np.empty((1, size, 1))
    x_test = np.empty((1, size, 1))
    y_test = np.empty((1, size, 1))

    for station in station_list:

        print(station)
        x_train = np.append(x_train, np.load(os.path.join(fld, 'x_train_{}.npy'.format(station))), axis=0)
        y_train = np.append(y_train, np.load(os.path.join(fld, 'y_train_{}.npy'.format(station))), axis=0)
        x_test = np.append(x_test, np.load(os.path.join(fld, 'x_val_{}.npy'.format(station))), axis=0)
        y_test = np.append(y_test, np.load(os.path.join(fld, 'y_val_{}.npy'.format(station))), axis=0)

    return x_train, y_train, x_test, y_test


def tcn_model_builder(d, k, s, f, max_len):
    model = dilated_tcn(num_feat=1,
                         num_classes=1,
                         nb_filters=f,
                         kernel_size=k,
                         dilations=d,
                         nb_stacks=s,
                         max_len=max_len,
                         regression=True)

    str_d = str(d).replace('[', '').replace(']', '').replace(',', 'x').replace(' ', '')
    param_str = str(f) + 'f_' + str(k) + 'k_' + str(s) + 's_' + str_d + 'd'
    return model, param_str


def predict_full(model, x, win_len, overlap = 200*40):
    orig_len = x.shape[1]
    stride = win_len - overlap
    nb_batch = int(x.shape[1]/stride) + 1
    nb_batch*stride+overlap
    x = np.append(x, np.zeros((1,nb_batch*stride+overlap-orig_len,1)), axis=1)
    y = np.zeros(x.shape)
    for i in reversed(range(nb_batch)):
        print(i)
        y[:,i*stride:(i*stride)+win_len,:] = model.predict(x[:,i*stride:(i*stride)+win_len,:])
    return y[:,:orig_len,:]


def xcorr_target(y, target):
    return np.flip(np.correlate(target, y, mode='same'), axis=0)


def confusion(PRED_cat, TRUE_cat, min_snr, threshold, total_length, search_win, detect=None):
    if detect == 1:
        TRUE_cat = TRUE_cat.loc[(TRUE_cat['STA'].str[-2:] == '31')]
    elif detect == 0:
        TRUE_cat = TRUE_cat.loc[(TRUE_cat['STA'].str[-2:] == 'AR')]
    print('len', len(TRUE_cat))
    my_true = TRUE_cat.loc[(TRUE_cat['SNR'] > min_snr)]
    my_pred = PRED_cat.loc[(PRED_cat['PVAL'] > threshold)]

    T = int((total_length / search_win))
    P = len(my_true)
    N = T - P
    TP = np.sum(my_true.PVAL >= threshold)
    # TP  = np.sum(my_pred.Y  > 0)
    FN = np.sum(my_true.PVAL < threshold)
    FP = np.sum(my_pred.Y == 0)
    TN = N - FP
    Ps = FP + TP
    Ns = TN + FN
    PPV = TP / Ps
    TPR = TP / P
    FPR = FP / N

    cm = np.array([[TN, FP], [FN, TP]])
    ex_cm = np.array([[TN, FP, N], [FN, TP, P], [Ns, Ps, T]])
    MAE = np.mean(np.abs(my_true.loc[(my_true['PVAL'] >= threshold)].ERR))
    ACC = (TP + TN) / (P + N)

    print(cm)
    print('precision:', PPV)
    print('recall/sensitivity:', TPR)
    print('specificity:', FPR)
    print('error:', MAE)
    print('acc:', ACC)

    return ex_cm, TPR, FPR, PPV, MAE, ACC


def catalog_predictions(test_cat, y_hat_filt, y_test_filt, alert_func, search_win, fs=40):
    print('Length of Test:', len(y_hat_filt))
    print('#ofPeaks:', int(len(y_test_filt)/(2*search_win*30)))
    aw = int(len(alert_func)/2)
    TRUE_cat = test_cat.copy()
    TRUE_cat['PVAL'] = ''
    TRUE_cat['ERR'] = ''

    y_peaks = sig.argrelextrema(y_hat_filt, np.greater)[0]
    cutoff = np.sort(np.take(y_hat_filt, y_peaks))[-int(len(y_test_filt)/(2*search_win*30))]
    y_peaks = [y for y in y_peaks if y_hat_filt[y] > cutoff]
    pmax = np.max(xcorr_target(alert_func, alert_func))
    min_thresh = cutoff / pmax

    PRED_cat = pd.DataFrame()
    PRED_cat = PRED_cat.assign(IDX=pd.Series(y_peaks).values)
    PRED_cat['PVAL'] = 0
    PRED_cat['Y'] = 0

    for i, event in TRUE_cat.iterrows():
        IDX = int(event.IDX + aw)
        SNR = event.SNR
        PLOC = np.argmax(y_hat_filt[IDX - search_win:IDX + search_win]) - search_win
        if np.abs(PLOC) == search_win:
            # print('miss:',IDX)
            # print(y_hat_filt[PLOC+search_win])
            PMAX = 0
        else:
            PMAX = y_hat_filt[IDX + PLOC]
        TRUE_cat.loc[i, 'PVAL'] = PMAX / pmax
        TRUE_cat.loc[i, 'ERR'] = PLOC / fs

    for i, event in PRED_cat.iterrows():
        IDX = int(event.IDX)
        P = max(y_test_filt[IDX - search_win:IDX + search_win])
        PRED_cat.loc[i, 'PVAL'] = y_hat_filt[IDX] / pmax
        PRED_cat.loc[i, 'Y'] = int(P > pmax)

    return TRUE_cat, PRED_cat, min_thresh


def getROCdata(PRED_cat, TRUE_cat, min_snr, max_snr, total_length, search_win, min_thresh, detect=None):
    if detect == 1:
        TRUE_cat = TRUE_cat.loc[(TRUE_cat['STA'].str[-2:] == '31')]
    elif detect == 0:
        TRUE_cat = TRUE_cat.loc[(TRUE_cat['STA'].str[-2:] == 'AR')]

    my_true = TRUE_cat.loc[(TRUE_cat['SNR'] > min_snr) & (TRUE_cat['SNR'] < max_snr)]

    thresholds = np.logspace(np.log10(min_thresh), .1, 50)
    col_names = ["TH", "TP", "FP", "TN", "FN", "TPR", "FPR", "PPV", "MAE", "ACC", "F"]
    ROC = pd.DataFrame(columns=col_names)

    for idx, TH in enumerate(thresholds):
        T = int((total_length / search_win))
        P = len(my_true)
        N = T - P
        TP = np.sum(my_true.PVAL >= TH)
        # TP  = np.sum(PRED_cat.loc[(PRED_cat['PVAL'] > TH)].Y  > 0)
        FP = np.sum(PRED_cat.loc[(PRED_cat['PVAL'] > TH)].Y == 0)
        TN = N - FP
        FN = np.sum(my_true.PVAL < TH)
        P_ = FP + TP
        N_ = TN + FN

        TPR = TP / P
        FPR = FP / N
        ACC = (TP + TN) / (P + N)
        PPV = TP / P_
        F = 2 * PPV * TPR / (PPV + TPR)
        MAE = np.mean(np.abs(my_true.loc[(my_true.PVAL >= TH)].ERR))

        ROC_row = pd.DataFrame([[TH, TP, FP, TN, FN, TPR, FPR, PPV, MAE, ACC, F]], columns=col_names, index=[idx])
        ROC = pd.concat([ROC, ROC_row])

    return ROC.apply(pd.to_numeric)


def model_parameters(model_name):
    dat_dir = 'data/' + model_name.split('--')[0][11:] + '/'
    mod_params = model_name.split('--')[1][:-3].split('_')
    f = int(mod_params[1][:-1])
    k = int(mod_params[2][:-1])
    s = int(mod_params[3][:-1])
    dd = mod_params[4][:-1]
    d = [int(ii) for ii in mod_params[4][:-1].split('x')]

    data_params = model_name.split('--')[0].split('_')
    aw = int(data_params[5][:-2])
    dec = float(data_params[6][:-3])
    amp = int(data_params[7][:-3])

    if aw < 500:
        alert_func = target_exp(aw, amp, dec)
    else:
        alert_func = target_burr(aw, amp, dec)

    return dat_dir, f, k, s, d, dd, aw, alert_func, amp


def model_tester(testset_name, max_len, search_win, model_name, results_file, size_lim=0, saveROC=False, saveCAT=False, model_dir='models/', fpr=.01):
    # Setup Results file (CSV)
    if saveCAT | saveROC:
        PLOT_dir = 'models/' + model_name[:3]
        if not os.path.exists(PLOT_dir):
            os.makedirs(PLOT_dir)

    if os.path.isfile(results_file):
        mod_params_df = pd.read_csv(results_file, index_col=0)
    else:
        mod_params_df = pd.DataFrame(
            columns=['NAME', 'DATASET', 'TPR', 'FPR', 'PPV', 'MAE', 'ACC', 'TPR1', 'FPR1', 'PPV1', 'MAE1', 'ACC1'])
    start_row = len(mod_params_df)
    complete_models = mod_params_df.NAME.values

    # Load Testset into Memory
    dat_dir, f, k, s, d, dd, aw, alert_func, amp = model_parameters(model_name)
    x_test = np.load(dat_dir + 'x_test_' + testset_name + '.npy')
    y_test = np.load(dat_dir + 'y_test_' + testset_name + '.npy')
    test_cat = pd.read_csv(dat_dir + 'test_cat_' + testset_name + '.csv')
    test_cat['REPORTER'] = test_cat['REPORTER'].astype('str')

    # Shorten Testset (if Desired)
    if size_lim > 0:
        x_test = x_test[:, :size_lim, :]
        y_test = y_test[:, :size_lim, :]
        test_cat = test_cat.loc[test_cat['IDX'] < size_lim]

    print('loading model...')
    tcn_model = dilated_tcn(num_feat=1,
                            num_classes=1,
                            nb_filters=f,
                            kernel_size=k,
                            dilations=d,
                            nb_stacks=s,
                            max_len=max_len,
                            regression=True)
    tcn_model.load_weights(os.path.join(model_dir, model_name))

    print('applying model to test set...')
    y_hat = predict_full(tcn_model, x_test, max_len)

    print('filtering output...')
    y_hat_filt = xcorr_target(y_hat[0, :, 0], alert_func)
    y_test_filt = xcorr_target(y_test[0, :, 0], alert_func)

    print('building prediction catalog...')
    TRUE_cat, PRED_cat, min_thresh = catalog_predictions(test_cat, y_hat_filt, y_test_filt, alert_func, search_win)

    print('printing confusion matrices...')
    ROC0 = getROCdata(PRED_cat, TRUE_cat, -999, 999, len(y_test_filt), search_win, min_thresh)
    ROC1 = getROCdata(PRED_cat, TRUE_cat, -999, 999, len(y_test_filt), search_win, min_thresh, 1)
    ROC = ROC0.loc[ROC0.FPR < fpr].reset_index().iloc[0]
    cm, TPR, FPR, PPV, MAE, ACC = confusion(PRED_cat, TRUE_cat, -50, ROC.TH, len(y_test_filt), search_win)
    cm1, TPR1, FPR1, PPV1, MAE1, ACC1 = confusion(PRED_cat, TRUE_cat, -50, ROC.TH, len(y_test_filt), search_win, 1)
    mod_params_df.loc[start_row] = [model_name, testset_name, TPR, FPR, PPV, MAE, ACC, TPR1, FPR1, PPV1, MAE1, ACC1]
    mod_params_df.to_csv(results_file)

    if saveCAT:
        TRUE_cat['DET'] = np.greater(TRUE_cat.PVAL, ROC.TH)
        CAT_file = "TRUECAT--{0}--{1}.csv".format(model_name, testset_name)
        CAT_filename = os.path.join(PLOT_dir, CAT_file)
        TRUE_cat.to_csv(CAT_filename)
        CAT_file = "PREDCAT--{0}--{1}.csv".format(model_name, testset_name)
        CAT_filename = os.path.join(PLOT_dir, CAT_file)
        PRED_cat.to_csv(CAT_filename)

    if saveROC:
        ROC0_file = "ROCCURVE--{0}--{1}.csv".format(model_name, testset_name)
        ROC0_filename = os.path.join(PLOT_dir, ROC0_file)
        ROC0.to_csv(ROC0_filename)

        ROC1_file = "ROCCURVE--{0}--{1}.csv".format(model_name, testset_name[:-2] + '31')
        ROC1_filename = os.path.join(PLOT_dir, ROC1_file)
        ROC1.to_csv(ROC1_filename)

        plotROC([ROC0_filename, ROC1_filename])


def plotROC(ROC_files, savefig=False, plt_dir = 'plots/', ax=None):
    ax = ax or plt.gca()
    ax.cla()

    for ROC_file in ROC_files:
        ROC_label = ROC_file.split('--')[-1].split('.')[0]
        if ROC_label[-2:] == 'AR':
            linestyle = '-'
        if ROC_label[-2:] == '31':
            linestyle = '--'
        else:
            linestyle = '-'
        ROC = pd.read_csv(ROC_file, index_col=0)
        ax.plot(ROC.FPR, ROC.TPR, label=ROC_label, linestyle=linestyle)

    ax.set_ylim((0, 1.1))
    ax.legend(loc='lower right')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))
    #plt.show()

    if savefig:
        if not os.path.exists(plt_dir):
            os.makedirs(plt_dir)
        plt_name = os.path.basename(ROC_file)[:-4] + '.png'
        plt_file = os.path.join(plt_dir, plt_name)
        print('saving', plt_file)
        plt.gcf().savefig(plt_file, transparent=True)


def evalPicks_old(cat, picks, search_win, fs=40):

    TRUE_cat = cat.copy()
    TRUE_cat['ERR'] = ''
    TRUE_cat['Y'] = 0

    PRED_cat = pd.DataFrame()
    PRED_cat = PRED_cat.assign(IDX=pd.Series(picks).values)
    PRED_cat['Y'] = 0

    for i, event in TRUE_cat.iterrows():
        IDX = int(event.IDX)
        errs = np.abs(IDX - picks)
        ERR = errs[errs.argmin()]
        if ERR < search_win:
            Y = 1
        else:
            Y = 0

        TRUE_cat.loc[i, 'Y'] = Y
        TRUE_cat.loc[i, 'ERR'] = ERR / fs

    for i, event in PRED_cat.iterrows():
        IDX = int(event.IDX)
        errs = np.abs(IDX - TRUE_cat.IDX.values)
        ERR = errs[errs.argmin()]
        if ERR < search_win:
            Y = 1
        else:
            Y = 0

        PRED_cat.loc[i, 'Y'] = Y

    return TRUE_cat, PRED_cat


def pick_err(eval_cat, ref_cat, search_win, fs):
    eval_cat['ERR'] = ''
    eval_cat['Y'] = 0

    for i, event in eval_cat.iterrows():
        IDX = int(event.IDX)
        errs = np.abs(IDX - ref_cat.IDX.values)
        ERR = errs[errs.argmin()]
        if ERR < search_win:
            Y = 1
        else:
            Y = 0

        eval_cat.loc[i, 'Y'] = Y
        eval_cat.loc[i, 'ERR'] = ERR / fs

    return eval_cat


def evalPicks(cat, picks, name, RESULTS_dir, size_lim, search_win, fs=40, runtime=0):
    # Setup Results Files
    eval_stats = pd.DataFrame(columns=['PICKER', 'TPR', 'FPR', 'F', 'PPV', 'MAE', 'ACC', 'TP', 'TN', 'FP', 'FN', 'RUNTIME'])
    eval_cat = cat.copy()

    if os.path.isfile(os.path.join(RESULTS_dir, 'EVAL_stats')):
        eval_stats = pd.read_csv(os.path.join(RESULTS_dir, 'EVAL_stats'), index_col=0)
    if os.path.isfile(os.path.join(RESULTS_dir, 'EVAL_cat')):
        eval_cat = pd.read_csv(os.path.join(RESULTS_dir, 'EVAL_cat'), index_col=0)

    # Compute TRUE_cat, PRED_cat and Stats
    TRUE_cat = cat.copy()
    PRED_cat = pd.DataFrame().assign(IDX=pd.Series(picks).values)

    TRUE_cat = pick_err(TRUE_cat, PRED_cat, search_win, fs)
    PRED_cat = pick_err(PRED_cat, TRUE_cat, search_win, fs)
    TPR, FPR, F, PPV, MAE, ACC, TP, TN, FP, FN = confusion_matrix(TRUE_cat, PRED_cat, size_lim / search_win)
    eval_stats.loc[len(eval_stats)] = [name, TPR, FPR, F, PPV, MAE, ACC, TP, TN, FP, FN, runtime]
    eval_cat[f'DET_{name}'] = TRUE_cat.Y

    # Save Results Files
    TRUE_cat.to_csv(os.path.join(RESULTS_dir, f'TRUE_cat_{name}'))
    PRED_cat.to_csv(os.path.join(RESULTS_dir, f'PRED_cat_{name}'))
    eval_stats.to_csv(os.path.join(RESULTS_dir, 'EVAL_stats'))
    eval_cat.to_csv(os.path.join(RESULTS_dir, 'EVAL_cat'))

    return eval_cat, eval_stats


def plotSNRPERF(cat, SNR_bins, det_columns, ax=None, sz=[None, None, None]):
    ax = ax or plt.gca()
    ax.cla()

    detperSNR = np.zeros((len(SNR_bins) - 1, len(det_columns)))
    # ax.plot(SNR_bins[:-1], np.ones(len(detperSNR)), linestyle='-', color='black')
    for j, det_column in enumerate(det_columns):
        for i in reversed(range(len(detperSNR))):
            low = SNR_bins[i]
            hi = SNR_bins[i + 1]

            SNRbin_cat = cat.loc[(cat.SNR > low) & (cat.SNR < hi)]
            detperSNR[i, j] = np.mean(SNRbin_cat[det_column])

        if j == 0:
            linestyle = '-'
            linewidth = '3'
        else:
            linestyle = '--'

        ax.plot(SNR_bins[:-1], detperSNR[:, j], linestyle=linestyle, label=det_column)

    ax.tick_params(labelsize=sz[2])
    ax.set_xlabel('Seismogram SNR (dB)', fontsize=sz[1])
    ax.set_ylabel('Detection Rate (%)', fontsize=sz[1])
    ax.set_yticks(np.arange(0, 1.1, step=0.1))
    ax.set_xticks(np.arange(-10, 43, step=5))
    ax.grid(color='grey', linestyle='--', linewidth=.5)
    ax.legend(loc='lower left', fontsize=sz[2])
    ax.invert_xaxis()


def get_y_peaks(y_hat_filt, alert_func, search_win):
    aw = int(len(alert_func) / 2)
    print('Length of Test:', len(y_hat_filt))
    print('#ofPeaks:', int(len(y_hat_filt) / (2 * search_win * 30)))
    y_peaks = sig.argrelextrema(y_hat_filt, np.greater)[0]
    cutoff = np.sort(np.take(y_hat_filt, y_peaks))[-int(len(y_hat_filt) / (2 * search_win * 30))]
    y_peaks = np.array([y for y in y_peaks if y_hat_filt[y] > cutoff])
    y_probs = np.take(y_hat_filt, y_peaks)
    return y_peaks - aw, y_probs


def catalog_predictions2(test_cat, y_peaks, y_probs, y_test_filt, search_win, fs=40):
    TRUE_cat = test_cat.copy()
    TRUE_cat['PVAL'] = ''
    TRUE_cat['ERR'] = ''

    pmax = np.max(y_probs)
    min_thresh = np.min(y_probs) / np.max(y_probs)

    PRED_cat = pd.DataFrame()
    PRED_cat = PRED_cat.assign(IDX=pd.Series(y_peaks).values)
    PRED_cat = PRED_cat.assign(PVAL=pd.Series(y_probs / pmax).values)
    PRED_cat['Y'] = 0

    for i, event in TRUE_cat.iterrows():
        IDX = int(event.IDX)
        SNR = event.SNR

        errs = np.abs(IDX - y_peaks)
        nearest_peak = errs.argmin()
        ERR = errs[nearest_peak]
        if ERR < 2 * search_win:
            PVAL = y_probs[nearest_peak]
        else:
            PVAL = 0

        TRUE_cat.loc[i, 'PVAL'] = PVAL / pmax
        TRUE_cat.loc[i, 'ERR'] = ERR / fs

    for i, event in PRED_cat.iterrows():
        IDX = int(event.IDX)
        P = max(y_test_filt[IDX - search_win:IDX + search_win])
        PRED_cat.loc[i, 'Y'] = int(P > pmax)

    return TRUE_cat, PRED_cat, min_thresh


def GetPicks_AIC(x_test, TH, run_samps=1000000, fs=40, return_CFn=True):
    picks = []
    CFn = []

    picker = aicdpicker.AICDPicker(t_ma=30, nsigma=TH, \
                                   t_up=2, nr_len=2, nr_coeff=.05, pol_len=10, pol_coeff=10, uncert_coeff=3)

    tot_samps = len(x_test)
    x_test = obspy.core.trace.Trace(data=x_test, header={"sampling_rate": fs}).filter('bandpass', freqmin=1, freqmax=4)

    for i in range(np.int(np.ceil(tot_samps / run_samps))):
        my_tr = obspy.core.trace.Trace(data=x_test[run_samps * (i):min(len(x_test), run_samps * (i + 1))],
                                       header={"sampling_rate": 40})
        scnl, pick, polarity, snr, uncert = picker.picks(my_tr)
        picks.extend([int((p - UTCDateTime(0)) * fs + run_samps * i) for p in pick])
        if return_CFn:
            CFn.extend(list(aicdpicker.AICDSummary(picker, my_tr).thres))
        else:
            print(i)

    return np.array(picks), np.array(CFn)


def GetPicks_KT(x_test, TH, run_samps=1000000, fs=40, return_CFn=True):
    picks = []
    CFn = []

    picker = ktpicker.KTPicker(t_win=5, t_ma=30, nsigma=TH,
                               t_up=2, nr_len=2, nr_coeff=.05, pol_len=10, pol_coeff=10, uncert_coeff=3)

    tot_samps = len(x_test)
    x_test = obspy.core.trace.Trace(data=x_test, header={"sampling_rate": fs}).filter('bandpass', freqmin=1, freqmax=4)

    for i in range(np.int(np.ceil(tot_samps / run_samps))):
        my_tr = obspy.core.trace.Trace(data=x_test[run_samps * (i):min(len(x_test), run_samps * (i + 1))],
                                       header={"sampling_rate": 40})
        scnl, pick, polarity, snr, uncert = picker.picks(my_tr)
        picks.extend([int((p - UTCDateTime(0)) * fs + run_samps * i) for p in pick])
        if return_CFn:
            CFn.extend(list(ktpicker.KTSummary(picker, my_tr).thres))
        else:
            print(i)

    return np.array(picks), np.array(CFn)


def GetPicks_FB_multi(x_test, TH, run_samps=1000000, fs=40, cores=2):
    tot_samps = len(x_test)
    tot_iters = np.int(np.ceil(tot_samps / run_samps))
    x_test = obspy.core.trace.Trace(data=x_test, header={"sampling_rate": fs})

    picker = fbpicker.FBPicker(t_long=5, freqmin=1, mode='rms', t_ma=30, nsigma=TH, \
                               t_up=2, nr_len=2, nr_coeff=.05, pol_len=10, pol_coeff=10, uncert_coeff=3)
    pick = []
    q = Queue()

    def little_picker(picker, tr, offset, fs=40):
        _, pick_list, _, _, _ = picker.picks(my_tr)
        pick_list = [int((p - UTCDateTime(0)) * fs + offset) for p in pick_list]
        q.put(pick_list)

    i = 0
    while i < tot_iters:
        processes = []
        for core in range(min(cores, tot_iters - i)):
            current_idx = run_samps * (i)
            next_idx = run_samps * (i + 1)
            my_tr = obspy.core.trace.Trace(data=x_test[current_idx:min(tot_samps, next_idx)],
                                           header={"sampling_rate": fs})

            t = multiprocessing.Process(target=little_picker, args=(picker, my_tr, current_idx))
            processes.append(t)
            t.start()
            print(i)
            i = i + 1

        for one_process in processes:
            one_process.join()

        while not q.empty():
            pick.extend(q.get())

    return np.array(pick)


def GetPicks_FB(x_test, TH, run_samps=1000000, fs=40, return_CFn=True):
    picks = []
    CFn = []

    picker = fbpicker.FBPicker(t_long=5, freqmin=1, mode='rms', t_ma=30, nsigma=TH, \
                               t_up=2, nr_len=2, nr_coeff=.05, pol_len=10, pol_coeff=10, uncert_coeff=3)

    tot_samps = len(x_test)
    x_test = obspy.core.trace.Trace(data=x_test, header={"sampling_rate": fs})

    for i in range(np.int(np.ceil(tot_samps / run_samps))):
        my_tr = obspy.core.trace.Trace(data=x_test[run_samps * (i):min(len(x_test), run_samps * (i + 1))],
                                       header={"sampling_rate": 40})
        scnl, pick, polarity, snr, uncert = picker.picks(my_tr)
        picks.extend([int((p - UTCDateTime(0)) * fs + run_samps * i) for p in pick])
        if return_CFn:
            CFn.extend(list(fbpicker.FBSummary(picker, my_tr).thres))
        else:
            print(i)

    return np.array(picks), np.array(CFn)


def GetPicks_DP(CFn, TH=100, return_CFn=True):
    y_peaks = sig.argrelextrema(CFn, np.greater)[0]
    y_peaks = np.array([y for y in y_peaks if CFn[y] > TH])
    y_peaks = eliminate_dups(y_peaks, 2 * 40)
    return y_peaks, CFn


def GetPicks_RF(x_test, cat, st_IDX=0, fs=40, freqmin=1, freqmax=4, return_CFn=True):
    if return_CFn:
        CFn = obspy.core.trace.Trace(data=x_test, header={"sampling_rate": fs}).filter('bandpass', freqmin=1, freqmax=4)
    else:
        CFn = x_test
    en_IDX = st_IDX + len(x_test)
    y_peaks = cat.loc[(cat.IDX >= st_IDX) & (cat.IDX <= en_IDX)].IDX - st_IDX
    return y_peaks, CFn


def plotPicks(picks, CFn, buffer=1500, fs=40, ax=None, labels=[None, None], sz=[None, None, None],
              colors=['blue', 'red']):
    ax = ax or plt.gca()
    ax.cla()
    t = np.linspace(0, len(CFn[buffer:-buffer]) - 1, len(CFn[buffer:-buffer])) / fs
    ax.plot(t, CFn[buffer:-buffer], color=colors[0])
    picks = [(p - buffer) / fs for p in picks if (p > buffer) & (p < len(CFn) - buffer)]

    ii, jj = ax.get_ylim()
    try:
        ax.vlines(picks, ii, jj, color=colors[1], lw=2)
    except:
        pass


def confusion_matrix(my_true, my_pred, T):
    #T = int((len(my_x) / search_win))
    P = len(my_true)
    N = int(T - P)
    TP = np.sum(my_true.Y == 1)
    FN = np.sum(my_true.Y == 0)
    FP = np.sum(my_pred.Y == 0)
    TN = N - FP
    P_ = FP + TP
    N_ = TN + FN

    TPR = TP / P
    FPR = FP / N
    ACC = (TP + TN) / (P + N)
    PPV = TP / P_
    F = 2 * PPV * TPR / (PPV + TPR)
    MAE = np.mean(np.abs(my_true.loc[(my_true.Y == 1)].ERR))
    return TPR, FPR, F, PPV, MAE, ACC, TP, TN, FP, FN


def eliminate_dups(arr, win):
    if arr.size > 0:
        new_arr = [arr[0]]
        for a in arr:
            if np.abs(new_arr[-1] - a) >= win:
                new_arr.append(a)
    else:
        new_arr = arr
    return np.array(new_arr)


def model_eval(testset_name, model_name, TH, search_win, max_len, size_lim=40*60*60*24*365, fs=40):
    TH_DP, TH_FB, TH_KT, TH_AIC = TH

    # Load Testset into Memory
    print('loading dataset...')
    dat_dir, f, k, s, d, dd, aw, alert_func, amp = model_parameters(model_name)
    x_test = np.load(dat_dir + 'x_test_' + testset_name + '.npy')
    size_lim = min(size_lim, x_test.shape[1])
    test_cat = pd.read_csv(dat_dir + 'test_cat_' + testset_name + '.csv')
    test_cat['REPORTER'] = test_cat['REPORTER'].astype('str')
    x_test = x_test[:, :size_lim, :]
    test_cat = test_cat.loc[test_cat['IDX'] < size_lim]

    # Setup Results files (CSV)
    RESULTS_dir = f'results/{testset_name}_{model_name[:-4]}'
    if not os.path.exists(RESULTS_dir):
        os.makedirs(RESULTS_dir)

    if TH_DP > 0:
        st = time.time()
        eval_name = f'DP_{TH_DP}'
        print('loading model...')
        tcn_model = dilated_tcn(num_feat=1,
                                num_classes=1,
                                nb_filters=f,
                                kernel_size=k,
                                dilations=d,
                                nb_stacks=s,
                                max_len=max_len,
                                regression=True)
        tcn_model.load_weights(os.path.join('models', model_name))

        print('implementing TCN-based picker...')
        y_hat = predict_full(tcn_model, x_test, max_len)
        y_hat_filt_tcn = np.roll(xcorr_target(y_hat[0, :, 0], alert_func), -aw)
        picks, _ = GetPicks_DP(y_hat_filt_tcn, TH=TH_DP, return_CFn=False)
        _, _ = evalPicks(test_cat, picks, eval_name, RESULTS_dir, size_lim, search_win, fs=fs, runtime=time.time()-st)

    if TH_FB > 0:
        st = time.time()
        eval_name = f'FB_{TH_FB}'
        print('implementing frequency-based picker...')
        picks = GetPicks_FB_multi(x_test[0,:,0], TH=TH_FB, run_samps=10000000, fs=40, cores=10)
        #picks, _ = GetPicks_FB(x_test[0,:,0], TH=TH_FB, return_CFn=False)
        _, _ = evalPicks(test_cat, picks, eval_name, RESULTS_dir, size_lim, search_win, fs=fs, runtime=time.time()-st)

    if TH_KT > 0:
        st = time.time()
        eval_name = f'KT_{TH_KT}'
        print('implementing kurtosis-based picker...')
        picks, _ = GetPicks_KT(x_test[0,:,0], TH=TH_KT, return_CFn=False)
        _, _ = evalPicks(test_cat, picks, eval_name, RESULTS_dir, size_lim, search_win, fs=fs, runtime=time.time()-st)

    if TH_AIC > 0:
        st = time.time()
        eval_name = f'AIC_{TH_AIC}'
        print('implementing AIC-based picker...')
        picks, _ = GetPicks_AIC(x_test[0,:,0], TH=TH_AIC, return_CFn=False)
        _, _ = evalPicks(test_cat, picks, eval_name, RESULTS_dir, size_lim, search_win, fs=fs, runtime=time.time()-st)
