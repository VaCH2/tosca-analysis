import pickle
import os
import sys
import pandas as pd 



configObject = pickle.load(open(f'smell=im_exout=False_excor=False_pca=True_exspr=True_braycurtis_specclu_None', 'rb'))
print(configObject.df)
#pickle.dump(scoreDict, open(os.path.join(temp_folder, f'MCCandPrecisionFor{iters}iters{subSample}percent'), 'wb'))  