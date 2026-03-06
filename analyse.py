from pathlib import Path

data_dir = Path("data/raw")
classes = ["Mario", "Luigi"]

for cls in classes:
    folder = data_dir / cls
    count = len(list(folder.glob("*.*")))
    print(f"{cls} : {count} images")
