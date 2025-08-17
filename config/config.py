from pathlib import Path

# --- DIRECTORY CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True) # Create artifacts dir if it doesn't exist

# --- DATA FILE PATHS ---
SOURCE_DATASET_PATH = DATA_DIR / "conference_reviews_dataset.csv"

# --- ARTIFACT FILE PATHS ---
CLUSTERING_ARTIFACT_PATH = ARTIFACTS_DIR / "user_clustering.pkl"
EMBEDDINGS_ARTIFACT_PATH = ARTIFACTS_DIR / "user_embeddings.pkl"

# Model-specific artifact paths
PEARSON_SIMILARITY_PATH = ARTIFACTS_DIR / "similarity_pearson.pkl"
PEARSON_PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions_pearson.pkl"

MUTIFACTOR_SIMILARITY_PATH = ARTIFACTS_DIR / "similarity_mutifactor.pkl"
MUTIFACTOR_PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions_mutifactor.pkl"

# --- MODEL PARAMETERS ---
# Clustering
NUM_CLUSTERS = 10

# Similarity
NUM_INFLUENCERS = 50
MUTIFACTOR_ALPHA = 0.01 # Temporal decay factor

# Prediction
NUM_NEIGHBORS = 25

# --- GEMINI SERVICE CONFIGURATION ---
GEMINI_ENV_PREFIX = "GEMINI_API_KEY_"
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_TASK_TYPE = "RETRIEVAL_DOCUMENT"
EMBEDDING_DIM = 768
EMBEDDING_BATCH_SIZE = 50