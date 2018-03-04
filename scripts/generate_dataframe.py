import argparse
import os
import glob
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--dirpath', required=True)
args = parser.parse_args()

# dirpath = "/home/hardian_lawi/plant-seedlings-classification/datasets/train"
dirpath = args.dirpath

df = {}
train_data = True
df["file"] = []
if train_data:
    df["species"] = []
    for dirname in os.listdir(dirpath):
        for filename in glob.glob(os.path.join(dirpath, dirname, "*")):
            df["file"].append(filename)
            df["species"].append(dirname)
    df = pd.DataFrame(df)
    df["species_id"] = df.species.map(dict(zip(df.species.unique(), range(df.species.nunique()))))
    df[["file", "species", "species_id"]].to_csv("../datasets/train.csv", index=False)
