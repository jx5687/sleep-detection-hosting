import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ukuran gambar
img_width, img_height = 64, 64

# Dataset path
train_data_dir = "dataset"

# Preprocessing gambar
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Data training
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Data validasi
validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Model CNN
model = Sequential()

model.add(Conv2D(
    32,
    (3,3),
    activation='relu',
    input_shape=(64,64,3)
))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(
    64,
    (3,3),
    activation='relu'
))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(
    units=128,
    activation='relu'
))

model.add(Dense(
    units=1,
    activation='sigmoid'
))

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Training model
model.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator
)

# Save model
model.save("sleep_detection_model.h5")

print("MODEL BERHASIL DISIMPAN!")