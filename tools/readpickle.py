import pandas as pd

dir = '/home/cyril.cero/catchall-dataset/cvat/cam8/MMdet/dev_mmdet2.pkl'

object = pd.read_pickle(dir)

print(object)