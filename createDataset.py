import os
import glob
import pandas as pd
import numpy as np

path = "dataArucoTags/" # use your path
all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent

df_from_each_file = (pd.read_csv(f) for f in all_files)
concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)

concatenated_df = concatenated_df.drop(['angleActual'],axis =1) # dropping extra col
# randomizing dataset
concatenated_df = concatenated_df.iloc[np.random.permutation(len(concatenated_df))]     
concatenated_df = concatenated_df.reset_index(drop=True)
print(concatenated_df)

concatenated_df.to_csv("RegressionDatasets/dataset_all.csv", index = False)

# creating new dataset with only 2 tahs values
df_small = concatenated_df.drop(["up_x", "up_y", "up_z", "down_x", "down_y", "down_z"], axis = 1)
df_small = df_small.reset_index(drop = True)
print(df_small)
df_small.to_csv("RegressionDatasets/dataset_2.csv", index = False)

