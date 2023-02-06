# RealTabFormer

The REaLTabFormer offers a unified framework for synthesizing tabular data of different types. A sequence-to-sequence model is used for generating synthetic relational datasets. The REaLTabFormer model for a non-relational tabular data uses GPT-2, and can be used out-of-the-box to model any tabular data with independent observations.

## Installation

REaLTabFormer is available on PyPi and can be easily installed with [pip](https://pypi.org/project/pip/) (Python version >= 3.7):

```bash
pip install realtabformer
```

## Usage

We show examples of using the REaLTabFormer for modeling and generating synthetic data from a trained model.

### REaLTabFormer for regular tabular data


```Python
# pip install realtabformer
import pandas as pd
from realtabformer import REaLTabFormer

df = pd.read_csv("foo.csv")

# NOTE: Remove any unique identifiers in the
# data that you don't want to be modeled.

# Non-relational or parent table.
rtf_model = REaLTabFormer(
    model_type="tabular",
    gradient_accumulation_steps=4,
    logging_steps=100)

# Fit the model on the dataset.
# Additional parameters can be
# passed to the `.fit` method.
rtf_model.fit(df)

# Save the model to the current directory.
# A new directory `rtf_model/` will be created.
# In it, a directory with the model's
# experiment id `idXXXX` will also be created
# where the artefacts of the model will be stored.
rtf_model.save("rtf_model/")

# Generate synthetic data with the same
# number of observations as the real dataset.
samples = rtf_model.sample(n_samples=len(df))

# Load the saved model. The directory to the
# experiment must be provided.
rtf_model2 = REaLTabFormer.load_from_dir(
    path="rtf_model/idXXXX")
```


### REaLTabFormer for relational data

```Python
# pip install realtabformer
import os
import pandas as pd
from realtabformer import REaLTabFormer

parent_df = pd.read_csv("foo.csv")
child_df = pd.read_csv("bar.csv")
join_on = "unique_id"

# Make sure that the key columns in both the
# parent and the child table have the same name.
assert ((join_on in parent_df.columns) and
        (join_on in child_df.columns))

# Non-relational or parent table. Don't include the
# unique_id field.
parent_model = REaLTabFormer(model_type="tabular")
parent_model.fit(parent_df.drop(join_on, axis=1))

pdir = Path("rtf_parent/")
parent_model.save(pdir)

# # Get the most recently saved parent model,
# # or a specify some other saved model.
# parent_model_path = pdir / "idXXX"
parent_model_path = sorted([
    p for p in pdir.glob("id*") if p.is_dir()],
    key=os.path.getmtime)[-1]

child_model = REaLTabFormer(
    model_type="relational",
    parent_realtabformer_path=parent_model_path,
    output_max_length=None,
    train_size=0.8)

child_model.fit(
    df=child_df,
    in_df=parent_df,
    join_on=join_on)

# Generate parent samples.
parent_samples = parent_model.sample(len(parend_df))

# Create the unique ids based on the index.
parent_samples.index.name = join_on
parent_samples = parent_samples.reset_index()

# Generate the relational observations.
child_samples = child_model.sample(
    input_unique_ids=parent_samples[join_on],
    input_df=parent_samples.drop(join_on, axis=1),
    gen_batch=64)

```
