import pickle
import os
import sys
import pandas as pd 

from utils import scale_df
from toscametrics import calculator


def main():
    """
    This script calculates source code measurements upon the provided TOSCA blueprint,
    passes it to Gaussian Mixture Model and returns the script being smelly or sound
    regarding the three smells 'Duplicate Block', 'Too many Attributes', and 'Insufficient Modularization'.
    """
    try:
        path = sys.argv[1]
    except Exception as e:
        print(e)
        print('Provide a valid TOSCA blueprint and execute the script by: python <path>/clusteringdetector.py <path-to-blueprint>')

    metrics = calculator.MetricCalculator([path], 'tosca_and_general').getresults
    value = metrics[path]
    df = pd.io.json.json_normalize(value, sep='_')
    value = df.to_dict(orient='records')[0]
    df = pd.DataFrame.from_dict({path : value}).T
    df = df.drop(['ttb_check', 'tdb_check', 'tob_check'], axis=1)

    configObject = pickle.load(open('models/config.pkl', 'rb'))
    principalComponents = configObject.principalComponents
    array = scale_df(df)
    array = principalComponents.transform(array)

    model = configObject.model
    label = model.predict(array)

    if label == 0:
        print('The provided blueprint is: Sound!')
    if label == 1:
        print('The provided blueprint is: Smelly..')

if __name__ == '__main__':
    main()