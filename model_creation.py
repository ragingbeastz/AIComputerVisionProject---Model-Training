import pandas as pd
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

#Read Dataset
df = pd.read_csv("C:\\Users\\Dimithri\\Documents\\Model Training Computer Vision\\car_labels.csv")

# Build image paths
datasetDirectory = "C:\\Users\\Dimithri\\Documents\\AlteredDataset"
df["full_path"] = df["path"].apply(lambda p: os.path.join(datasetDirectory, p.replace("\\", "/")))


# Split into train/val/test sets
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42) 
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Create label index mapping
labelNames = sorted(df["label"].unique())
labelIndexes = {name: i for i, name in enumerate(labelNames)}

# Apply label indices
train_df["label_idx"] = train_df["label"].map(labelIndexes)
val_df["label_idx"] = val_df["label"].map(labelIndexes)
test_df["label_idx"] = test_df["label"].map(labelIndexes)


img_size = (300, 300)
batch_size = 32

# Function to create TensorFlow dataset from DataFrame
def build_dataset(df, shuffle=True):
    paths = df["full_path"].values
    labels = df["label_idx"].values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def process(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, img_size)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    ds = ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = build_dataset(train_df)
val_ds = build_dataset(val_df, shuffle=False)
test_ds = build_dataset(test_df, shuffle=False)

# Create the model
base_model = EfficientNetB3(
    weights='imagenet',
    include_top=False,
    input_shape=(300, 300, 3)  # match your current setup
)

base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
output = Dense(len(labelNames), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training model...")
model.fit(train_ds, validation_data=val_ds, epochs=10)

print("Evaluating model...")
model.evaluate(test_ds)

# Save the model
model.save("car_classifier_model.keras")

