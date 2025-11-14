import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# ✅ Dataset paths
train_dir = r"D:\unified\ASL_detection\ASL_detection\asl_alphabet_train"
test_dir = r"D:\unified\ASL_detection\ASL_detection\asl_alphabet_test"

# ✅ Image Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# ✅ Load pretrained mobilenetv2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# ✅ Add custom layers
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(29, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ✅ Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ Train the model
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)

# ✅ Save model
model.save("asl_vgg16_model.h5")
print("✅ Model training complete. Saved as asl_vgg16_model.h5")
