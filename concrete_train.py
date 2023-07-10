import pycaret.regression as carreg
from time import perf_counter
import pandas as pd
import pickle as pkl
from pycaret.utils import enable_colab


# How many models we will tune
n_select = 6

# Read in the training data
train_df = pd.read_csv("train_concrete.csv", index_col=False)  ## Your code here

# Set up the regression experiment session
print("*** Setting up session***")
t1 = perf_counter()
exp1 = carreg.setup(data=train_df, target="csMPa")  ## Your code here
t2 = perf_counter()
print(f"*** Set up: {t2 - t1:.2f} seconds")

# Do a basic comparison of models (no turbo)
best = carreg.compare_models(n_select=n_select, turbo=False)  ## Your code here
t3 = perf_counter()
print(f"*** compare_models: {t3 - t2:.2f} seconds")

# List the best models
print(f"*** Best:")
for b in best:
    print(f"\t{b.__class__.__name__}")

# Go through the list of models
output_result = []
for i, model in enumerate(best):
    print(f"I = {i}. and model = {model}")
    # Tune the model (try 24 parameter combinations)
    ## Your code here
    print(f"*** {i} - {model.__class__.__name__}     ***")
    model_created = carreg.create_model(best[i])
    tuned_model = carreg.tune_model(model_created, n_iter=24)

    results = carreg.pull()
    output_result.append(results)

    # Finalize the model
    final_model = carreg.finalize_model(tuned_model)

    ## Your code here

    # Save the model
    ## Your code here
    carreg.save_model(final_model, model.__class__.__name__)

print()

t4 = perf_counter()
print(f"*** Tuning and finalizing: {t4 - t3:.2f} seconds")
print(f"*** Total time: {t4 - t1:.2f} seconds")
