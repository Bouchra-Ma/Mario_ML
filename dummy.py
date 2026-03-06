from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix

from CHARGEMENT import load_data   # <-- importer X et y

# Charger les données
X, y = load_data()

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Dummy model : prédit toujours la classe majoritaire
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)

# prédictions
y_dummy = dummy.predict(X_test)

print("=== Dummy Model ===")
print(classification_report(y_test, y_dummy))
print(confusion_matrix(y_test, y_dummy))
