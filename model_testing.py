# =========================
# Env + imports
# =========================
import os, json, csv
os.environ["TF_DETERMINISTIC_OPS"] = "0"

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input as effnetPreprocess
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, BackupAndRestore

# =========================
# Paths / config
# =========================
runDir       = "runs/car_make_v1"
feBackupDir  = os.path.join(runDir, "fe_backup")
ftBackupDir  = os.path.join(runDir, "ft_backup")
ckptPath     = os.path.join(runDir, "best_model.ckpt")

csvPath     = r"C:\Users\Dimithri\Documents\Model Training Computer Vision\car_labelsV4.csv"
imgRoot     = r"C:\Users\Dimithri\Documents\MiniDatasetV4"
modelPath   = "model_car_make.keras"
labelsPath  = "labels_car_make.csv"

os.makedirs(runDir, exist_ok=True)

seed = 42
tf.keras.utils.set_random_seed(seed)

gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", len(gpus))
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except:
        pass

try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision:", mixed_precision.global_policy())
except Exception:
    pass

# =========================
# Data
# =========================
df = pd.read_csv(csvPath)
df["label"] = df["label"].apply(lambda s: s.split("_")[0])
df["full_path"] = df["path"].apply(lambda p: os.path.join(imgRoot, p.replace("\\", "/")))
df = df[df["full_path"].apply(os.path.exists)].reset_index(True)

labelNames = sorted(df["label"].unique())
labelIndex = {name: i for i, name in enumerate(labelNames)}
df["label_idx"] = df["label"].map(labelIndex).astype("int32")

trainDf, testDf = train_test_split(df, test_size=0.2, random_state=seed)
trainDf, valDf  = train_test_split(trainDf, test_size=0.15, random_state=seed)

numClasses = len(labelNames)
imgSize    = (300, 300)
batchSize  = 16
AUTOTUNE   = tf.data.AUTOTUNE

dataAug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.08),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
    tf.keras.layers.RandomContrast(0.1),
], name="aug")

def buildDataset(frame: pd.DataFrame, training: bool):
    paths  = frame["full_path"].values
    labels = frame["label_idx"].values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(pth, lab):
        img = tf.io.read_file(pth)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, imgSize)
        img = tf.cast(img, tf.float32)
        img = effnetPreprocess(img)
        return img, lab

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(lambda x, y: (dataAug(x, training=True), y), num_parallel_calls=AUTOTUNE)
        ds = ds.shuffle(2048, seed=seed)
    return ds.batch(batchSize).prefetch(AUTOTUNE)

trainDs = buildDataset(trainDf, training=True)
valDs   = buildDataset(valDf,   training=False)
testDs  = buildDataset(testDf,  training=False)

# =========================
# Model
# =========================
inputs = Input(shape=(imgSize[0], imgSize[1], 3))
base   = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inputs)
base.trainable = False

x = GlobalAveragePooling2D(name="avg_pool")(base.output)
x = Dropout(0.0, name="dropout")(x)
outputs = Dense(numClasses, activation="softmax", dtype="float32", name="pred")(x)
model = Model(inputs, outputs, name="EffNetB3_finetune")

top5Metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)

def hasBackup(dirPath: str) -> bool:
    return tf.io.gfile.exists(os.path.join(dirPath, "chief", "checkpoint"))

def unfreezeNonBN():
    for lyr in base.layers:
        if not isinstance(lyr, tf.keras.layers.BatchNormalization):
            lyr.trainable = True

callbacksFe = [
    BackupAndRestore(backup_dir=feBackupDir),
    EarlyStopping(monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6),
    ModelCheckpoint(filepath=ckptPath, save_weights_only=True,
                    monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
]

callbacksFt = [
    BackupAndRestore(backup_dir=ftBackupDir),
    EarlyStopping(monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True),
    ModelCheckpoint(filepath=ckptPath, save_weights_only=True,
                    monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
]

optimizerFe = Adam(learning_rate=3e-4)
lossFe      = tf.keras.losses.SparseCategoricalCrossentropy()

stepsPerEpoch = int(np.ceil(len(trainDf) / batchSize))
lrSchedule    = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=1e-5,
    first_decay_steps=5 * stepsPerEpoch,
    t_mul=2.0, m_mul=0.9, alpha=1e-6
)
optimizerFt = Adam(learning_rate=lrSchedule, clipnorm=1.0)
lossFt      = tf.keras.losses.SparseCategoricalCrossentropy()

epochsFe = 3
epochsFt = 30

feHasBackup = hasBackup(feBackupDir)
ftHasBackup = hasBackup(ftBackupDir)
hasBestFe   = tf.io.gfile.exists(ckptPath + ".index")

if ftHasBackup:
    print("Resume FT")
    unfreezeNonBN()
    model.compile(optimizer=optimizerFt, loss=lossFt, metrics=["accuracy", top5Metric])
    model.fit(trainDs, validation_data=valDs, epochs=epochsFt, callbacks=callbacksFt, verbose=1)

elif feHasBackup:
    print("Resume FE")
    model.compile(optimizer=optimizerFe, loss=lossFe, metrics=["accuracy", top5Metric])
    model.fit(trainDs, validation_data=valDs, epochs=epochsFe, callbacks=callbacksFe, verbose=1)
    print("Start FT")
    if tf.io.gfile.exists(ckptPath + ".index"):
        model.load_weights(ckptPath)
    unfreezeNonBN()
    model.compile(optimizer=optimizerFt, loss=lossFt, metrics=["accuracy", top5Metric])
    model.fit(trainDs, validation_data=valDs, epochs=epochsFt, callbacks=callbacksFt, verbose=1)

elif hasBestFe:
    print("Skip FE → FT")
    model.load_weights(ckptPath)
    unfreezeNonBN()
    model.compile(optimizer=optimizerFt, loss=lossFt, metrics=["accuracy", top5Metric])
    model.fit(trainDs, validation_data=valDs, epochs=epochsFt, callbacks=callbacksFt, verbose=1)

else:
    print("Start FE")
    model.compile(optimizer=optimizerFe, loss=lossFe, metrics=["accuracy", top5Metric])
    model.fit(trainDs, validation_data=valDs, epochs=epochsFe, callbacks=callbacksFe, verbose=1)
    print("Start FT")
    if tf.io.gfile.exists(ckptPath + ".index"):
        model.load_weights(ckptPath)
    unfreezeNonBN()
    model.compile(optimizer=optimizerFt, loss=lossFt, metrics=["accuracy", top5Metric])
    model.fit(trainDs, validation_data=valDs, epochs=epochsFt, callbacks=callbacksFt, verbose=1)

if tf.io.gfile.exists(ckptPath + ".index"):
    model.load_weights(ckptPath)

print("Eval (test)")
testMetrics = model.evaluate(testDs, verbose=1)
print("Test:", dict(zip(model.metrics_names, testMetrics)))

print("Saving model")
model.save(modelPath, include_optimizer=False)

print("Saving labels CSV")
with open(labelsPath, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["label", "index"])
    for k, v in labelIndex.items():
        w.writerow([k, int(v)])

print("Done")
