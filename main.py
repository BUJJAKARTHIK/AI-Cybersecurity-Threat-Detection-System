from src.preprocessing import load_data, preprocess_data
from src.model import train_model, evaluate_model
from src.detection import detect_threats
from src.visualization import plot_results

# Load dataset
data = load_data("data/KDDTrain+.txt")

# Preprocess
X_train, X_test, y_train, y_test = preprocess_data(data)

# Train model
model = train_model(X_train, y_train)

# Evaluate model
evaluate_model(model, X_test, y_test)

# Detect threats
predictions = detect_threats(model, X_test)

# Visualization
plot_results(y_test, predictions)

from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))