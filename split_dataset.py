import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Dossiers
raw_dir = Path("data/raw")
output_dir = Path("data")

classes = ["Mario", "Luigi"]

# Pourcentage
test_size = 0.15
val_size = 0.15

# Création des dossiers
for split in ["train", "val", "test"]:
    for cls in classes:
        (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

# Split par classe
for cls in classes:
    images = list((raw_dir / cls).glob("*.*"))

    # Train + temp (val+test)
    train_imgs, temp_imgs = train_test_split(
        images, test_size=test_size + val_size, random_state=42
    )

    # Val + test
    val_imgs, test_imgs = train_test_split(
        temp_imgs, test_size=test_size / (test_size + val_size), random_state=42
    )

    # Copie des fichiers
    for img in train_imgs:
        shutil.copy(img, output_dir / "train" / cls)

    for img in val_imgs:
        shutil.copy(img, output_dir / "val" / cls)

    for img in test_imgs:
        shutil.copy(img, output_dir / "test" / cls)

print("Split terminé !")
