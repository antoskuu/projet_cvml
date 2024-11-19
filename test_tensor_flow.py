import tensorflow as tf
import os
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt
import sys
import io
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Chemins vers les dossiers d'images et de labels
images_dir = 'C:/Users/anton/Dev/ms-coco/ms-coco/images/train/train-resized'
labels_dir = 'C:/Users/anton/Dev/ms-coco/ms-coco/labels/train/train'
images_dir_test = 'C:/Users/anton/Dev/ms-coco/ms-coco/images/test/test-resized'
yaml_file = 'C:/Users/anton/Dev/ms-coco/ms-coco/coco.yaml'

def yaml_to_list(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as f:
        categories = yaml.safe_load(f)['names']
        categories_list = list(categories.values())
    return categories_list

def parse_image_and_label(image_path, label_path, target_size=(224, 224), num_classes=80):
    # Charger et prétraiter l'image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = tf.image.random_flip_left_right(img)  # Flip horizontal aléatoire
    img = tf.image.random_brightness(img, max_delta=0.1)  # Ajustement de la luminosité aléatoire
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)  # Ajustement du contraste aléatoire
    img = tf.cast(img, tf.float32) / 255.0  # Normalisation

    # Charger les labels et les convertir en one-hot encoding
    label_vector = np.zeros(num_classes)
    with open(label_path, 'r', encoding='utf-8') as f:
        for label in f.readlines():
            label_idx = int(label.strip())
            label_vector[label_idx] = 1

    return img, label_vector

def load_data(images_dir, labels_dir, sample_size=10, target_size=(224, 224), num_classes=80):
    image_files = os.listdir(images_dir)
    sample_image_files = random.sample(image_files, sample_size)

    images = []
    all_labels = []

    for image_file in sample_image_files:
        image_path = os.path.join(images_dir, image_file)
        label_file = image_file.replace('.jpg', '.cls')
        label_path = os.path.join(labels_dir, label_file)

        if os.path.exists(label_path):
            img, label_vector = parse_image_and_label(image_path, label_path, target_size, num_classes)
            images.append(img)
            all_labels.append(label_vector)
        else:
            print(f"Label file not found for image: {image_file}")

    x = tf.stack(images)  # Convertit la liste d'images en tenseur
    y = tf.convert_to_tensor(all_labels, dtype=tf.float32)

    return x, y

def print_image_categories(image_index, y_train, categories):
    label_vector = y_train[image_index]
    category_indices = np.where(label_vector == 1)[0]
    category_names = [categories[i] for i in category_indices]
    return category_names

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0  # Normalisation
    return img

def predictions_to_categories(predictions, categories, top_k=5):
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_categories = [(categories[i], predictions[i]) for i in top_indices]
    return top_categories

categories = yaml_to_list(yaml_file)
num_classes = len(categories)

# Charger les données d'entraînement
x_train, y_train = load_data(images_dir, labels_dir, sample_size=5000, num_classes=num_classes)

# Définir le modèle
base_model = tf.keras.applications.VGG16(input_shape=(224, 224, 3),
                                         include_top=False,
                                         weights='imagenet')

base_model.trainable = False  # Geler les couches du modèle de base

model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Ajouter du dropout pour la régularisation
    tf.keras.layers.Dense(num_classes, activation='sigmoid')  # Utiliser 'sigmoid' pour multi-label classification
])
model.summary()

# Compiler le modèle
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Utiliser 'binary_crossentropy' pour multi-label classification
              metrics=['accuracy'])

# Entraîner le modèle avec validation
history = model.fit(x_train, y_train, epochs=2, batch_size=32, validation_split=0.2)

# Sauvegarder le modèle
model.save('model.h5')

# Évaluer le modèle sur l'ensemble de validation
val_loss, val_accuracy = model.evaluate(x_train, y_train)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# Visualiser les courbes de perte et de précision
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.show()

# Charger les images de test
test_image_files = os.listdir(images_dir_test)
predictions_dict = {}

for image_file in test_image_files:
    image_path = os.path.join(images_dir_test, image_file)
    test_image = load_and_preprocess_image(image_path)
    test_image = np.expand_dims(test_image, axis=0)  # Ajouter une dimension pour le batch

    # Faire une prédiction
    predictions = model.predict(test_image)
    predicted_indices = np.where(predictions[0] > 0.5)[0]  # Seuil de 0.5 pour la classification multi-label

    # Ajouter les prédictions au dictionnaire
    image_id = os.path.splitext(image_file)[0]
    predictions_dict[image_id] = predicted_indices.tolist()

# Sauvegarder les prédictions dans un fichier JSON
with open('predictions.json', 'w') as json_file:
    json.dump(predictions_dict, json_file, indent=4)

print("Predictions saved to predictions.json")