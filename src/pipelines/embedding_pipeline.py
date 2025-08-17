import pandas as pd
import numpy as np
import os
import time
from itertools import cycle
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm

import config.config as config  # Import centralized configuration
from src.utils.file_handlers import load_pickle, save_pickle # Import utilities

# Load environment variables from .env file
load_dotenv()

# --- Gemini Service (Self-contained within this pipeline file) ---

class KeyManager:
    """Manages and cycles through a list of Gemini API keys."""
    def __init__(self, env_prefix: str):
        self.keys = [v for k, v in os.environ.items() if k.startswith(env_prefix)]
        if not self.keys:
            raise ValueError(f"❌ No API keys found with prefix '{env_prefix}' in .env file.")
        self._key_cycler = cycle(self.keys)
        print(f"🔑 KeyManager initialized with {len(self.keys)} API keys.")

    def get_next_key(self) -> str:
        """Returns the next API key in the cycle."""
        return next(self._key_cycler)

class GeminiService:
    """A service class for handling Gemini API embedding operations."""
    def __init__(self, env_prefix: str, model: str, task_type: str, dim: int):
        try:
            from google import genai
            from google.genai import types
            from google.api_core import exceptions as google_exceptions
            self.genai = genai
            self.types = types
            self.google_exceptions = google_exceptions
        except ImportError:
            raise ImportError("❌ Please install google-generativeai: pip install google-generativeai")
            
        self.key_manager = KeyManager(env_prefix)
        self.model = model
        self.task_type = task_type
        self.output_dim = dim

    @staticmethod
    def _normalize_embedding(embedding: list) -> list:
        """Normalizes the embedding to a unit vector."""
        np_embedding = np.array(embedding)
        norm = np.linalg.norm(np_embedding)
        return (np_embedding / norm).tolist() if norm != 0 else embedding

    def embed_content(self, batch_docs: List[str]) -> List[List[float]] | None:
        """Generates embeddings for a batch of documents with retry logic."""
        retries, max_retries, backoff_delay = 0, 3, 5
        while retries < max_retries:
            api_key = self.key_manager.get_next_key()
            client = self.genai.Client(api_key=api_key)
            try:
                result = client.models.embed_content(
                    model=self.model,
                    contents=batch_docs,
                    config=self.types.EmbedContentConfig(task_type=self.task_type, output_dimensionality=self.output_dim)
                )
                return [self._normalize_embedding(e.values) for e in result.embeddings]
            except (self.google_exceptions.InternalServerError, self.google_exceptions.ResourceExhausted, self.google_exceptions.PermissionDenied) as e:
                retries += 1
                print(f"\n⚠️ Gemini API error (key ...{api_key[-4:]}): {type(e).__name__}. "
                      f"Retrying in {backoff_delay}s... ({retries}/{max_retries})")
                time.sleep(backoff_delay)
                backoff_delay *= 2
            except Exception as e:
                print(f"\n❌ An unexpected error occurred during embedding with key ...{api_key[-4:]}: {e}")
                break
        
        print(f"❌ Skipping batch after failing all retry attempts.")
        return None

# --- Main Pipeline Function ---

def run_embedding_pipeline():
    """
    Generates and saves vector embeddings for user reviews.
    This pipeline is recoverable; it loads existing embeddings and only
    generates embeddings for users that are missing.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        print("\n" + "="*60)
        print("--- 🚀 Running Step 2: User Review Embedding Pipeline ---")
        print("="*60)

        # --- 1. Load Existing Embeddings (Upsert Logic) ---
        print(f"📂 Checking for existing embeddings at '{config.EMBEDDINGS_ARTIFACT_PATH}'...")
        user_embeddings = load_pickle(config.EMBEDDINGS_ARTIFACT_PATH)
        if user_embeddings is None:
            user_embeddings = {} # Start fresh if file doesn't exist or fails to load

        # --- 2. Identify Missing Users ---
        print(f"📂 Loading source data from '{config.SOURCE_DATASET_PATH}' to identify users...")
        df = pd.read_csv(config.SOURCE_DATASET_PATH)
        if df.empty:
            print("❌ Error: Source data file is empty.")
            return False
            
        user_docs = df.groupby('user_id')['review'].apply(lambda reviews: ' '.join(reviews)).to_dict()
        
        existing_user_ids = set(user_embeddings.keys())
        all_user_ids = set(user_docs.keys())
        missing_user_ids = sorted(list(all_user_ids - existing_user_ids)) # Sort for deterministic order
        
        if not missing_user_ids:
            print("✅ All user embeddings are already present. Nothing to do.")
            print("\n--- ✅ Step 2: Embedding Pipeline Completed Successfully ---\n")
            return True
        
        print(f"   - Found {len(all_user_ids)} total users.")
        print(f"   - Found {len(existing_user_ids)} existing embeddings.")
        print(f"   - Found {len(missing_user_ids)} users requiring new embeddings.")

        # --- 3. Generate Embeddings for Missing Users ---
        print("🧠 Initializing Gemini Service and generating new embeddings...")
        gemini_service = GeminiService(
            env_prefix=config.GEMINI_ENV_PREFIX, 
            model=config.EMBEDDING_MODEL, 
            task_type=config.EMBEDDING_TASK_TYPE,
            dim=config.EMBEDDING_DIM
        )
        
        docs_to_embed = [user_docs[uid] for uid in missing_user_ids]
        new_embeddings = {}
        
        for i in tqdm(range(0, len(docs_to_embed), config.EMBEDDING_BATCH_SIZE), desc="Embedding Batches"):
            batch_ids = missing_user_ids[i:i+config.EMBEDDING_BATCH_SIZE]
            batch_docs = docs_to_embed[i:i+config.EMBEDDING_BATCH_SIZE]
            
            embeddings_result = gemini_service.embed_content(batch_docs)
            if embeddings_result:
                for user_id, emb in zip(batch_ids, embeddings_result):
                    new_embeddings[user_id] = emb
        
        if len(new_embeddings) != len(missing_user_ids):
            print(f"⚠️ Warning: Could not generate embeddings for all missing users. "
                  f"Successfully generated {len(new_embeddings)} out of {len(missing_user_ids)}.")

        # --- 4. Update and Save Artifacts ---
        print(f"💾 Updating and saving embeddings artifact to '{config.EMBEDDINGS_ARTIFACT_PATH}'...")
        user_embeddings.update(new_embeddings)
        save_pickle(user_embeddings, config.EMBEDDINGS_ARTIFACT_PATH)
        print(f"   - Total embeddings now in file: {len(user_embeddings)}")
        
        print("\n--- ✅ Step 2: Embedding Pipeline Completed Successfully ---\n")
        return True

    except Exception as e:
        print(f"❌ An unexpected error occurred in the embedding pipeline: {e}")
        return False

if __name__ == '__main__':
    # This allows the script to be run standalone for testing/debugging
    run_embedding_pipeline()