import os
import json
import argparse
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.applications.vgg19 import VGG19, preprocess_input
from sklearn.model_selection import StratifiedKFold
from util import f1_micro

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default="../output_data", type=str, help="path to output directory")
parser.add_argument("--dataset", default="../datasets/train.csv", type=str, help="path to the csv file generated using generate_dataframe.py")
parser.add_argument("--n_splits", default=5, type=int, help="Number of stratified k fold splits")
parser.add_argument("--batch_size", default=128, type=int, help="Size per training batch")
parser.add_argument("--epochs", default=10, type=int, help="number of epochs for training")

# Parse arguments
args = parser.parse_args()

output_dir = args.output_dir
dataset = args.dataset

log_file = os.path.join(output_dir, "log.json")
model_filepath = os.path.join(output_dir, "plant_vgg19.h5")
weights_filepath = os.path.join(output_dir, "plant_vgg19_weights.h5")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# No of split for skfold cross validation
n_splits = args.n_splits

# Training hyperparameters
batch_size = args.batch_size
epochs = args.epochs

# Read pre-generated dataset comprising of 3 columns (file, species, species_id)
df = pd.read_csv(dataset).sample(frac=1.0)

# number of classes
n_classes = df.species.nunique()

# Load and resize all images
print("Loading images...")
all_imgs = []
for filename in df.file:
    img = image.load_img(filename, target_size=(299, 299, 3))
    img = image.img_to_array(img)
    all_imgs.append(img)

# Convert to numpy array
X = np.array(all_imgs)  # Matrix of (m x 299 x 299 x 3)
X = preprocess_input(X)  # Preprocess using VGG19 preprocess_input
y = df.as_matrix(columns=["species_id"])  # Convert target to numpy array of m x 1

# Load model
# include_top is used to remove all the layers after block conv5
model = VGG19(weights="imagenet", include_top=False, input_shape=img.shape)

# Freeze all layers
for layer in model.layers:
    layer.trainable = False

# re-add the removed layers
x = model.output
x = Flatten(name="flatten")(x)
x = Dense(4096, activation="relu", name="fc1")(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation="relu", name="fc2")(x)
x = Dropout(0.5)(x)
x = Dense(n_classes, activation="softmax", name="predictions")(x)

# Redefine the model
model = Model(inputs=model.input, outputs=x, name="final_model")

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[f1_micro])

# Define a splitter
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

# Training
print("Start cross-validation training...")
histories = []
for train, val in skf.split(X, y):
    Xtrain = X[train, :]
    ytrain = to_categorical(y[train, :])
    Xval = X[val, :]
    yval = to_categorical(y[val, :])
    history = model.fit(Xtrain, ytrain, batch_size=batch_size, epochs=epochs, validation_data=(Xval, yval))
    histories.append(history)

# Full training
print("Full training...")
history = model.fit(X, y, batch_size=batch_size, epochs=epochs, shuffle=True)
histories.append(history)

print("Save whole model...")
model.save(model_filepath)

print("Save weights of the model")
model.save(weights_filepath)

with open(log_file, "w") as f:
    json.dump(histories)
