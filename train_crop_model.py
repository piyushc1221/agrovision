import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
import os

# === CONFIGURATION ===
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS_TOP = 10
EPOCHS_FINE = 5
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
MODEL_SAVE_PATH = "crop_model_vgg16.h5"

# === Load dataset ===
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Automatically detect number of classes
NUM_CLASSES = train_ds.element_spec[1].shape[1]
print(f"[INFO] Detected {NUM_CLASSES} classes:", train_ds.class_names)

# === Save class names to text file for Flask use ===
with open("class_names.txt", "w") as f:
    for cname in train_ds.class_names:
        f.write(cname + "\n")
print("[INFO] Saved class names to class_names.txt")

# === Normalize datasets (0-255 → 0-1) ===
def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(normalize_img)
val_ds = val_ds.map(normalize_img)

# === Data augmentation ===
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# === Build model ===
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # freeze base initially

# Custom classifier head
inputs = layers.Input(shape=(224,224,3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Phase 1: Train top layers ===
print("=== Training top layers ===")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_TOP)

# === Phase 2: Fine-tune last few layers ===
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("=== Fine-tuning last layers ===")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE)

# === Save the trained model ===
model.save(MODEL_SAVE_PATH)
print(f"[INFO] Model saved to {MODEL_SAVE_PATH}")
