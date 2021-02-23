import pandas as pd

def load_data(path):
    if path.split('.')[-1] == 'dat':
        df = pd.read_csv(path, sep='::', header=None, engine='python')
    else:
        df = pd.read_csv(path)
    return df