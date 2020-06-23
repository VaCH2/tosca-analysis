import pickle
import os
import sys
import pandas as pd 


x = 'im'
configObject = pickle.load(open(f'{x}Config', 'rb'))
pickle.dump(configObject.model, open(f'{x}Config_model', 'wb'))  