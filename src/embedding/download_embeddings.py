import os
import sys

def download_model(model_name="all-MiniLM-L6-v2"):
    print(f"Initializing download of {model_name}...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers not installed. Exiting.")
        sys.exit(1)
        
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib", "models", model_name))
    print(f"Saving model to {save_path}")
    
    # This automatically downloads from HF Hub if not available locally
    model = SentenceTransformer(model_name)
    model.save(save_path)
    print("Download and serialization complete.")

if __name__ == "__main__":
    download_model()
