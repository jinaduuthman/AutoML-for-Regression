import numpy as np
import pandas as pd
from scipy.stats import chi2

# Read in the data
mice_df = pd.read_csv("mice.csv")

# Figure out the possible gene types
gene_types = list(mice_df.gene_type.unique())
print(f"Possible gene types:{gene_types}")

## Your code here
# Get values of gene type J that has cancer.
J_has_cancer = mice_df.loc[
    (mice_df["gene_type"] == "J") & (mice_df["has_cancer"] == True)
]
J_has_cancer = J_has_cancer.shape[0]

# Get values of gene type J that has_no_cancer.
J_has_No_cancer = mice_df.loc[
    (mice_df["gene_type"] == "J") & (mice_df["has_cancer"] == False)
]
J_has_No_cancer = J_has_No_cancer.shape[0]


# Get values of gene type R that has cancer.
R_has_cancer = mice_df.loc[
    (mice_df["gene_type"] == "R") & (mice_df["has_cancer"] == True)
]
R_has_cancer = R_has_cancer.shape[0]


# Get values of gene type R that has_no_cancer.
R_has_No_cancer = mice_df.loc[
    (mice_df["gene_type"] == "R") & (mice_df["has_cancer"] == False)
]
R_has_No_cancer = R_has_No_cancer.shape[0]


# Get values of gene type K that has cancer.
K_has_cancer = mice_df.loc[
    (mice_df["gene_type"] == "K") & (mice_df["has_cancer"] == True)
]
K_has_cancer = K_has_cancer.shape[0]


# Get values of gene type K that has_no_cancer.
K_has_No_cancer = mice_df.loc[
    (mice_df["gene_type"] == "K") & (mice_df["has_cancer"] == False)
]
K_has_No_cancer = K_has_No_cancer.shape[0]


observed_matrix = np.array(
    [
        [R_has_No_cancer, R_has_cancer],
        [J_has_No_cancer, J_has_cancer],
        [K_has_No_cancer, K_has_cancer],
    ]
)
group_totals = np.sum(observed_matrix, axis=1)
cat_totals = np.sum(observed_matrix, axis=0)
expected_proportions = cat_totals / np.sum(cat_totals)
expected_matrix = np.outer(group_totals, expected_proportions)

print(f"Observed Matrix = {observed_matrix}")
print()
print(f"Expected Matrix = {expected_matrix}")
print()

numerator = np.square(observed_matrix - expected_matrix)
x2 = np.sum(numerator / expected_matrix)
print(f"X2 = {x2:.3f}")
print()

# The p-test
p_less = chi2.cdf(x2, 2)
p = 1.0 - p_less
print(f"p = {p}")
