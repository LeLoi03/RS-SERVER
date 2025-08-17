import pickle
from pathlib import Path

def save_pickle(data, path: Path):
    """Saves data to a pickle file."""
    try:
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ Data successfully saved to '{path}'")
    except Exception as e:
        print(f"❌ Error saving data to '{path}': {e}")

def load_pickle(path: Path):
    """Loads data from a pickle file."""
    if not path.exists():
        print(f"⚠️ File not found at '{path}', returning None.")
        return None
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ Data successfully loaded from '{path}'")
        return data
    except Exception as e:
        print(f"❌ Error loading data from '{path}': {e}")
        return None