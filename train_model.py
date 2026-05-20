from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam

# ================= DATASET PATH =================

train_dir = "train"
test_dir = "test"

# ================= IMAGE PREPROCESSING =================

train_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48,48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical"
)

# ================= CNN MODEL =================

model = Sequential()

# First Layer
model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(48,48,1)
    )
)

model.add(MaxPooling2D(2,2))

# Second Layer
model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)

model.add(MaxPooling2D(2,2))

# Third Layer
model.add(
    Conv2D(
        128,
        (3,3),
        activation='relu'
    )
)

model.add(MaxPooling2D(2,2))

# Flatten
model.add(Flatten())

# Dense Layer
model.add(Dense(256, activation='relu'))

# Dropout
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(7, activation='softmax'))

# ================= COMPILE MODEL =================

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ================= TRAIN MODEL =================
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam

# ================= DATASET PATH =================

train_dir = "train"
test_dir = "test"

# ================= IMAGE PREPROCESSING =================

train_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48,48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical"
)

# ================= CNN MODEL =================

model = Sequential()

# First Layer
model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(48,48,1)
    )
)

model.add(MaxPooling2D(2,2))

# Second Layer
model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)

model.add(MaxPooling2D(2,2))

# Third Layer
model.add(
    Conv2D(
        128,
        (3,3),
        activation='relu'
    )
)

model.add(MaxPooling2D(2,2))

# Flatten
model.add(Flatten())

# Dense Layer
model.add(Dense(256, activation='relu'))

# Dropout
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(7, activation='softmax'))

# ================= COMPILE MODEL =================

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ================= TRAIN MODEL =================

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=15
)

# ================= SAVE MODEL =================

model.save("emotion_model.h5")

print("✅ Model Saved Successfully")
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=15
)

# ================= SAVE MODEL =================

model.save("emotion_model.h5")

print("✅ Model Saved Successfully")