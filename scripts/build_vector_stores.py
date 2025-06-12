# scripts/build_vector_stores.py

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pymilvus import MilvusClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.side_character_app.vector_stores.builder import build_persona_vector_db   

def main():
    """Main function to build all persona vector stores."""
    # --- 1. Setup and Initialization ---
    load_dotenv()
    google_api_key = os.getenv("GEMINI_API_KEY")
    if not google_api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file.")

    # --- 2. Define Paths ---
    project_root = Path(__file__).resolve().parents[1]
    input_file = project_root / "data" / "processed" / "side_character_labeled_conversations.json"
    db_dir = project_root / "data" / "vector_stores"
    db_path = db_dir / "milvus_side_characters.db"

    # Create directory for the database if it doesn't exist
    db_dir.mkdir(exist_ok=True)

    # --- 3. Initialize Clients ---
    print(f"Initializing Milvus client at: {db_path}")
    client = MilvusClient(str(db_path))

    print("Initializing Google Gemini embedding function...")
    embedding_fn = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=google_api_key
    )

    # --- 4. Configuration (Generic Personas) ---
    # Using archetypes instead of character names
    PERSONA_CONFIG = {
        "Wise Mentor": {"collection_name": "wise_mentor_db"},
        "Comedic Relief": {"collection_name": "comedic_relief_db"},
        "Skeptical Realist": {"collection_name": "skeptical_realist_db"},
        "Loyal Sidekick": {"collection_name": "loyal_sidekick_db"}
    }
    MIN_CONFIDENCE = 8 # Set your desired confidence threshold

    # --- 5. Execution Loop ---
    print("\nStarting vector database build process...")
    for label, config in PERSONA_CONFIG.items():
        print(f"\n--- Building database for: {label} ---")
        build_persona_vector_db(
            client=client,
            embedding_fn=embedding_fn,
            json_path=str(input_file),
            collection_name=config["collection_name"],
            label=label,
            min_confidence=MIN_CONFIDENCE
        )

    print("\n✅ All persona vector databases have been successfully created.")

if __name__ == "__main__":
    main()