
import os, csv
import pandas as pd
import tensorflow as tf
import time
from datetime import timedelta
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models, optimizers


print(tf.config.list_physical_devices("GPU"))

makesDirectory = f"C:\\Users\\Dimithri\\Documents\\MiniDatasetV4\\"
accuracies = pd.DataFrame(columns=["make", "accuracy"])
times = []
progress = 0

if not os.path.exists("makes_left.csv"):
    with open("makes_left.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for make in os.listdir(makesDirectory):
            writer.writerow([make])


makesLeftDF = pd.read_csv("makes_left.csv", header=None, names=["make"])
totalMakes = len(makesLeftDF)

for idx, row in makesLeftDF.iterrows():
    currentMake = row["make"]
    currentTime = time.time()

    print(f"Training make:  {currentMake}")

    currentMake_path = f"C:\\Users\\Dimithri\\Documents\\MiniDatasetV4\\{currentMake}"
    rows = []

    print("Creating Labels...")
    for model in os.listdir(currentMake_path):
        model_path = os.path.join(currentMake_path, model)
        for year in os.listdir(model_path):
            year_path = os.path.join(model_path, year)
            label = f"{model}".replace(" ", "_")
            for image_file in os.listdir(year_path):
                image_path = os.path.join(currentMake_path, model, year, image_file)
                rows.append([image_path, label])

    print("Done Creating Labels.")

    df = pd.DataFrame(rows, columns=["path", "label"])
    print(df.head())


    MODEL_OUT  = f"model_car_model_{currentMake}.keras"
    LABELS_OUT = f"model_car_model_{currentMake}.csv"

    IMG_SIZE   = (300, 300)
    BATCH_SIZE = 16
    EPOCHS     = 15
    LR         = 3e-4
    SEED       = 42

    tf.keras.utils.set_random_seed(SEED)
    AUTOTUNE = tf.data.AUTOTUNE


    df["label"] = df["label"].apply(lambda s: s.split("_")[0])
    df["full_path"] = df["path"].apply(lambda p: os.path.join(currentMake_path, p.replace("\\", "/")))
    df = df[df["full_path"].apply(os.path.exists)].reset_index(drop=True)

    # Label encoding
    label_names = sorted(df["label"].unique())
    label_to_idx = {name: i for i, name in enumerate(label_names)}
    df["label_idx"] = df["label"].map(label_to_idx).astype("int32")
    num_classes = len(label_names)

    # Train/val/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label_idx"])
    train_df, val_df  = train_test_split(train_df, test_size=0.15, random_state=SEED, stratify=train_df["label_idx"])


    def make_ds(frame: pd.DataFrame, training: bool):
        paths  = frame["full_path"].values
        labels = frame["label_idx"].values
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        def _load(pth, y):
            img = tf.io.read_file(pth)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, IMG_SIZE)
            img = tf.cast(img, tf.float32)
            img = preprocess_input(img)  
            return img, y

        ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
        if training:
            ds = ds.shuffle(2048, seed=SEED)
        return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    train_ds = make_ds(train_df, training=True)
    val_ds   = make_ds(val_df,   training=False)
    test_ds  = make_ds(test_df,  training=False)


    base = EfficientNetB3(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3), pooling="avg")
    base.trainable = False  

    x = layers.Dropout(0.0)(base.output)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(base.input, out)

    model.compile(
        optimizer=optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)

    acc = float(model.evaluate(test_ds, verbose=0)[1])
    accuracies.loc[len(accuracies)] = {"make": currentMake, "accuracy": acc}


    print(f"Saving model {currentMake}")
    model.save(rf"C:\Users\Dimithri\Documents\Model Training Computer Vision\Models\model_car_model_{currentMake}.keras", include_optimizer=False)
    print(f"Saving labels CSV {currentMake}")
    with open(rf"C:\Users\Dimithri\Documents\Model Training Computer Vision\Models\labels_car_model_{currentMake}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "index"])
        for k, v in label_to_idx.items():
            w.writerow([k, int(v)])
    
    makesLeftDF = makesLeftDF[makesLeftDF["make"] != currentMake].reset_index(drop=True)
    makesLeftDF.to_csv("makes_left.csv", header=False, index=False)

    print(f"Completed make: {currentMake}")
    progress += 1
    times.append(time.time() - currentTime)
    

    avg = (sum(times) / len(times)) if times else 0.0
    remaining = max(totalMakes - progress, 0)
    eta_secs = int(round(avg * remaining))

    eta_str = str(timedelta(seconds=eta_secs)) if eta_secs > 0 else "0:00:00"
    print(f"ETA: {eta_str}")

    accuracies.to_csv("accuracies.csv", index=False, float_format="%.6f")



print("Done.")