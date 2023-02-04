import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

for pos in range(1,6):
    path = "errorAnalysisDatasets/"
    csv_files = glob.glob(os.path.join(path, f"*_{pos}.csv"))

    dfs = (pd.read_csv(f) for f in csv_files)
    concat_df =  pd.concat(dfs, ignore_index=True)
    concat_df["error"] = abs(concat_df["error"])
    error_df = concat_df.groupby("Original_Angle").mean()["error"]

    plt.plot(error_df, label = f"Position {pos}")

plt.xlabel("Original Angle")
plt.ylabel("Error in Estimation")    
plt.legend()
plt.xticks(np.arange(15,135, 15))
plt.show()

    
