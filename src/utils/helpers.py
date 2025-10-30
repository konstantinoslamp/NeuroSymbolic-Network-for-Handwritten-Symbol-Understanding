def log_message(message: str) -> None:
    """Logs a message to the console."""
    print(f"[LOG] {message}")

def validate_input(data: any, expected_type: type) -> bool:
    """Validates that the input data is of the expected type."""
    return isinstance(data, expected_type)

def save_model(model: any, filepath: str) -> None:
    """Saves the model to the specified filepath."""
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath: str) -> any:
    """Loads a model from the specified filepath."""
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)