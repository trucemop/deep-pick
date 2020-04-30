import os
from obspy.clients.fdsn import Client
from UTILS_TF2 import monthly_miniseed_downloader, monthly_catalog_downloader, merge_csv, snr_csv

#############################################################################################################
#  This python script will download all the data necessary to train a deep teleseismic signal detector.     #
#  To download the raw waveforms, the script utilizes ObsPy and queries the IRIS database                   #
#  To download the arrival catalogs, the script queries the ISC catalogs at: http://www.isc.ac.uk/          #
#############################################################################################################

# OPEN CONNECTION TO IRIS
c = Client("IRIS")
print(c)


# DEFINE STATIONS
stations = [['PDAR', 'PD31', 'IM'],
            ['MKAR', 'MK31', 'IM'],
            ['ABKAR', 'ABKAR', 'KZ'],
            ['ILAR', 'IL31', 'IM'],
            ['TXAR', 'TX31', 'IM'],
            ['BURAR', 'BURAR', 'RO'],
            ['ASAR', 'AS31', 'AU'],
            ['NVAR', 'NVAR', 'IM']]

stations = [['PDAR', 'PD31', 'IM'],
            ['MKAR', 'MK31', 'IM']]


# DEFINE TIMEFRAME
start_year = 2010
end_year = 2016


for station_name, station_code, network in stations:

    print('-' * 25, '\nDOWNLOADING DATA FOR ', station_name, '\n', '-' * 25, sep='')
    dat_dir = 'data/Data_' + station_name
    cat_dir = 'data/Cat_' + station_name

    # CREATE DIRECTORY AND DOWNLOAD FILES

    if not os.path.exists(dat_dir):
        os.makedirs(dat_dir)
    if not os.path.exists(cat_dir):
        os.makedirs(cat_dir)

    for yr in range(start_year, end_year):
        for mm in range(1, 13):

            # DOWNLOAD WAVEFORMS

            DAT_file = "{0}_{1:04}_{2:02}.mseed".format(station_name, yr, mm)
            DAT_filename = os.path.join(dat_dir, DAT_file)
            monthly_miniseed_downloader(c, yr, mm, DAT_filename, station_code, network)

            # DOWNLOAD FULL-ARRAY CATALOG

            CAT_file = "{0}_{1:04}_{2:02}_arrivals.csv".format(station_name, yr, mm)
            CAT_filename = os.path.join(cat_dir, CAT_file)
            monthly_catalog_downloader(yr, mm, station_name, CAT_filename)

            # DOWNLOAD SINGLE-TRACE CATALOG

            CAT_file = "{0}_{1:04}_{2:02}_arrivals.csv".format(station_code, yr, mm)
            CAT_filename = os.path.join(cat_dir, CAT_file)
            monthly_catalog_downloader(yr, mm, station_code, CAT_filename)

    # MERGE CATALOGS AND CALCULATE SNR

    FULL_CAT_file = '{}Arrivals.csv'.format(station_name)
    FULL_CAT_filename = os.path.join(cat_dir, FULL_CAT_file)
    merge_csv(cat_dir, FULL_CAT_filename)

    SNR_CAT_file = '{}ArrivalsSNR.csv'.format(station_name)
    SNR_CAT_filename = os.path.join(cat_dir, SNR_CAT_file)
    snr_csv(SNR_CAT_filename, FULL_CAT_filename, dat_dir)
