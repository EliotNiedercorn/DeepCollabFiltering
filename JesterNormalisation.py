import pandas as pd  # Used to read .xls file into pandas DataFrame and convert it to numpy matrix. In order to do this, one might need to install "xlrd" with "pip install xlrd"
import numpy as np

"""
We use here the Jester dataset [https://goldberg.berkeley.edu/jester-data/], it's a matrix under .xls format with
dimensions (24983 X 101). The file name is "jester-data-1.xls".
Each row corresponds to a user and each column to a joke. The matrix entries are ratings given as real values 
from -10.00 to +10.00. Here, 24.983 users have rated 36 or more jokes.
The value "99" corresponds to "null" = "not rated"
The first column gives the number of jokes rated by that user. The next 100 columns give the ratings for jokes 01 - 100.
"""

# Prepare the observed matrix with missing entries as the input data for the AutoEncoder.
xls2pandas = pd.read_excel('Data/jester-data-1.xls', header=None)  # Read the .xls file into a pandas DataFrame
pandas2numpy = xls2pandas.to_numpy()  # Convert the DataFrame to a numpy matrix
jester = pandas2numpy[:, 1:]  # Remove first column which corresponds to the number of jokes rated by given user (start at column 1 through the end)
print(jester)
jester_normalized = np.where(jester == 99.0, jester, (jester + 10) / 20)
print(jester_normalized)
jester_normalized[jester_normalized == 99] = np.nan

# Save the modified matrix as a new Excel file
modified_jester = pd.DataFrame(jester_normalized)  # Convert back
writer = pd.ExcelWriter('Data/jester_normalized.xlsx', engine='xlsxwriter')
modified_jester.to_excel(writer, index=False)
writer.save()
