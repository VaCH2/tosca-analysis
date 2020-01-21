from data import Data
from stats import Stats
from prep import Preprocessing

data = Data('tosca_and_general', 'all', 'topology')
data_stats = Stats(data)
prep = Preprocessing(data_stats, constants=False, corr=False)
