import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# ===============================
# CONFIGURATION
# ===============================
train_dir = "trial"  # path to dataset folder
img_size = (64, 64)
batch_size = 32
epochs = 5

# ===============================
# CHECK FOR EXISTING MODEL
# ===============================
if os.path.exists("asl_model.h5"):
    print("⚠️ Model already exists as 'asl_model.h5'. It will be overwritten.\n")

# ===============================
# DATA PREPARATION
# ===============================
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_data = train_gen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# ===============================
# MODEL CREATION
# ===============================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(64, 64, 3)
)
base_model.trainable = False  # freeze base layers for faster training

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(29, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# CALLBACKS
# ===============================
early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=2,
    restore_best_weights=True
)

# ===============================
# TRAINING
# ===============================
print("\n🚀 Training Started...\n")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[early_stop],
    verbose=1
)
print("\n✅ Training Completed!\n")

# ===============================
# EVALUATE & DISPLAY RESULTS
# ===============================
final_train_acc = history.history["accuracy"][-1]
final_val_acc = history.history["val_accuracy"][-1]

print(f"📊 Final Training Accuracy: {final_train_acc * 100:.2f}%")
print(f"📈 Final Validation Accuracy: {final_val_acc * 100:.2f}%")

# ===============================
# SAVE MODEL
# ===============================
model.save("asl_model.h5")
print("\n💾 Model saved successfully as 'asl_model.h5'.")
print("You can now run 'streamlit run app.py' to use it!\n")
