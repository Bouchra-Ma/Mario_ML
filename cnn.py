print("ok")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Dossiers
train_dir = "data/train"
val_dir = "data/val"
test_dir = "data/test"

# Générateurs
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir, target_size=(64, 64), batch_size=16, class_mode='binary'
)

val_data = val_gen.flow_from_directory(
    val_dir, target_size=(64, 64), batch_size=16, class_mode='binary'
)

test_data = test_gen.flow_from_directory(
    test_dir, target_size=(64, 64), batch_size=16, class_mode='binary'
)

# Modèle CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entraînement
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)

# Évaluation
loss, acc = model.evaluate(test_data)
print("Accuracy test :", acc)
