import tensorflow as tf
import os
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Chemins vers les dossiers d'images et de labels
images_dir = 'C:/Users/anton/Dev/ms-coco/ms-coco/images/train/train-resized'
labels_dir = 'C:/Users/anton/Dev/ms-coco/ms-coco/labels/train/train'

#images_dir_test = 'C:/Users/anton/Dev/ms-coco/ms-coco/images/test/test-resized'
#labels_dir_test = 'C:/Users/anton/Dev/ms-coco/ms-coco/labels/test/test'

test_image_path = 'C:/Users/anton/Dev/ms-coco/ms-coco/images/test/test-resized/000000091619.jpg'

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


# Pour vérifier que les labels attachés aux images d'entrainement sont bons
# print("\nPremiers labels:", y_train[0])
# categorie2 = print_image_categories(0, y_train, categories)
# plt.imshow(x_train[0])  # Affiche la première image
# plt.title(f"Labels: {categorie2}")
# plt.show()

# Définir le modèle
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='sigmoid')  # Utiliser 'sigmoid' pour multi-label classification
])
model.summary()

# Compiler le modèle
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Utiliser 'binary_crossentropy' pour multi-label classification
              metrics=['accuracy'])

# Entraîner le modèle
model.fit(x_train, y_train, epochs=10, batch_size=100)

# Sauvegarder le modèle
model.save('model.h5')


### Test du modèle avec l'image de test "test_image_path" ###
# Charger et prétraiter l'image de test
test_image = load_and_preprocess_image(test_image_path)
test_image = np.expand_dims(test_image, axis=0)  # Ajouter une dimension pour le batch

# Faire une prédiction
predictions = model.predict(test_image)
print("Predictions:", predictions)
its=predictions_to_categories(predictions[0], categories)
print("Categories:", its)

# Afficher les résultats
predicted_categories = print_image_categories(0, predictions, categories)
print("Predicted Categories:", predicted_categories)
plt.imshow(test_image[0])
plt.title(f"Prédictions: {its}")
plt.show()



