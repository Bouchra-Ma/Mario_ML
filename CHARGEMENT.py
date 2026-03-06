from pathlib import Path
import numpy as np
from PIL import Image

def load_data():
    data_dir = Path("data/raw")
    classes = ["Mario", "Luigi"]

    images = []
    labels = []

    for label, class_name in enumerate(classes):
        class_dir = data_dir / class_name
        print("Lecture du dossier :", class_dir)

        for img_path in class_dir.glob("*.*"):
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((64, 64))
                arr = np.array(img, dtype=np.float32) / 255.0
                images.append(arr.flatten())
                labels.append(label)
            except Exception as e:
                print(f"Erreur lors du chargement de {img_path}: {e}")

    X = np.stack(images)
    y = np.array(labels)

    return X, y
