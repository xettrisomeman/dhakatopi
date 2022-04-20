# type: ignore

import os


import pandas as pd


path = os.path.join(os.path.dirname(__file__), "../newsdatanepali/")


train_data = pd.read_csv(path+"train.csv")
test_data = pd.read_csv(path+"valid.csv")
