import numpy as np
import pandas as pd
import os


target_folder = "./datasets"
excel_files = os.listdir(target_folder)

row_nums = 0
for file in excel_files:
    if not file.endswith(".xlsx"):
        continue
    df = pd.read_excel(os.path.join(target_folder, file))
    print("The file `%s` has %d number of papers" % (file, df.shape[0]))
    row_nums += df.shape[0]
    
print("Total papers scrapped: %d" % row_nums)