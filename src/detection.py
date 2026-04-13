def detect_threats(model, X_test):
    predictions = model.predict(X_test)

    print("\nThreat Detection Results:\n")

    for i, pred in enumerate(predictions[:20]):
        if pred == 1:
            print(f"⚠️ Threat detected at index {i}")
        else:
            print(f"✅ Normal traffic at index {i}")

    return predictions