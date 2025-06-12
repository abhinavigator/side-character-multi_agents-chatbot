# scripts/test_retriever.py

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the src directory to the Python path to allow for absolute imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

# --- Imports from our app modules ---
# We import the tool function directly to test it
from src.side_character_app.app.tools import retrieve_persona_examples

# --- Imports from libraries ---
from pymilvus import MilvusClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def main():
    """
    A standalone script to test and demonstrate the RAG retriever's performance
    for each persona-specific vector database.
    """
    # --- 1. Setup and Initialization ---
    load_dotenv()
    google_api_key = os.getenv("GEMINI_API_KEY")
    if not google_api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file.")

    # --- 2. Define Paths ---
    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "data" / "vector_stores" / "milvus_side_characters.db"
    
    # --- 3. Initialize Clients ---
    print("Initializing clients...")
    client = MilvusClient(str(db_path))
    embedding_fn = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=google_api_key
    )
    print("Clients initialized successfully.\n")

    # --- 4. Define Persona Databases and Test Queries ---
    
    # This maps the archetype name to its corresponding Milvus collection name
    archetype_db_map = {
        "Wise Mentor": "wise_mentor_db",
        "Comedic Relief": "comedic_relief_db",
        "Skeptical Realist": "skeptical_realist_db",
        "Loyal Sidekick": "loyal_sidekick_db"
    }
    
    # A dictionary of test queries, one for each archetype
    test_queries = {
        "Wise Mentor": "I am struggling to find meaning in my work.",
        "Comedic Relief": "Tell me a funny story about a misunderstanding.",
        "Skeptical Realist": "My plan to start a new company is perfect and has no flaws.",
        "Loyal Sidekick": "I feel like I failed and let everyone down."
    }

    # --- 5. Main Test Loop ---
    
    for archetype, query in test_queries.items():
        print("="*80)
        print(f"Testing Retriever for Archetype: {archetype}")
        print(f"Test Query: '{query}'")
        print("="*80)
        
        # Get the collection name for the current archetype
        collection_name = archetype_db_map[archetype]
        
        # Call the retriever function directly
        retrieved_context = retrieve_persona_examples(
            query=query,
            collection_name=collection_name,
            client=client,
            embedding_fn=embedding_fn,
            archetype_name=archetype
        )
        
        # Print the formatted output that would be sent to the LLM
        print("--- Retrieved Context (Top 5) ---\n")
        print(retrieved_context)
        print("\n\n")

if __name__ == "__main__":
    main()