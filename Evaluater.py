from UTILS import model_eval
import os

GPU = 0

os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU)
print(f'GPU: {GPU}')

# Define Models, Testset and Results filenames
testsets = ['TXAR', 'PDAR', 'ILAR',  'BURAR', 'ABKAR', 'MKAR', 'ASAR']

models = [
    'best_model_360sec_200buf_0.02fmin_300aw_0.02dec_10amp_max--PDAR+MKAR+ABKAR+ASAR+BURAR+ILAR_20f_16k_12s_1x2x4x8d.h5',
    'best_model_360sec_200buf_0.02fmin_300aw_0.02dec_10amp_max--ASAR+MKAR+ABKAR+TXAR+BURAR+ILAR_20f_16k_12s_1x2x4x8d.h5',
    'best_model_360sec_200buf_0.02fmin_300aw_0.02dec_10amp_max--PDAR+MKAR+ABKAR+TXAR+BURAR+ASAR_20f_16k_12s_1x2x4x8d.h5',
    'best_model_360sec_200buf_0.02fmin_300aw_0.02dec_10amp_max--PDAR+MKAR+ABKAR+TXAR+ASAR+ILAR_20f_16k_12s_1x2x4x8d.h5',
    'best_model_360sec_200buf_0.02fmin_300aw_0.02dec_10amp_max--PDAR+MKAR+ABKAR+TXAR+ASAR+ILAR_20f_16k_12s_1x2x4x8d.h5',
    'best_model_360sec_200buf_0.02fmin_300aw_0.02dec_10amp_max--PDAR+ASAR+ABKAR+TXAR+BURAR+ILAR_20f_16k_12s_1x2x4x8d.h5',
    'best_model_360sec_200buf_0.02fmin_300aw_0.02dec_10amp_max--PDAR+MKAR+ABKAR+TXAR+BURAR+ILAR_20f_16k_12s_1x2x4x8d.h5']

TH = [[750,7,6,0], [500,7,6,0], [350,7,6,0], [700,7,6,0], [500,7,5,0], [500,7,6,0], [400,7,6,0]]
TH = [[87,5,0,0], [66,5,0,0], [100,5,0,0], [105,5,0,0], [150,5,0,0], [70,5,0,0], [80,5,0,0]]
TH = [[0,0,3,0], [0,0,3,0], [0,0,3,0], [0,0,3,0], [0,0,3,0], [0,0,3,0], [0,0,3,0]]


testsets = ['PDAR_2015_1_2015_2']
models = ['best_model_360sec_200buf_0.02fmin_300aw_0.02dec_10amp_max--MKAR_20f_16k_12s_1x2x4x8d.h5']
TH = [[500,0,0,0]]

# Define the limits for the test size (based on available system RAM)
max_len = 10000000  # number of samples to predict in single pass of model (limited by GPU RAM)
size_lim = max_len * 1  # overall number of samples to predict (limited by OS RAM)
# Define the size of the search window
search_win = 2 * 40

#for i in range(len(models)):
for i in [GPU]:
    model_name = models[i]
    testset_name = testsets[i]
    th = TH[i]
    print(model_name)
    print(testset_name)
    model_eval(testset_name, model_name, th, search_win, max_len)


