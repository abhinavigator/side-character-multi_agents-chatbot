# scripts/run_preprocessing.py

import os
import json
import sys

# Add the src directory to the Python path to allow for absolute imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.side_character_app.data_processing.loader import load_and_clean_data
from src.side_character_app.data_processing.builder import build_side_character_conversations

def main():
    """Main function to run the data preprocessing pipeline."""
    # --- Define Paths ---
    # Assumes the script is run from the project root directory
    project_root = os.path.dirname(os.path.dirname(__file__))
    raw_data_dir = os.path.join(project_root, "data", "raw")
    processed_data_dir = os.path.join(project_root, "data", "processed")
    output_path = os.path.join(processed_data_dir, "side_character_personas.json")

    # Create the processed data directory if it doesn't exist
    os.makedirs(processed_data_dir, exist_ok=True)

    # --- Run Pipeline ---
    # 1. Load data
    dataframes = load_and_clean_data(raw_data_dir)
    
    # 2. Build structured conversations
    results = build_side_character_conversations(dataframes)
    
    # 3. Filter results based on conversation count
    MAX_CONVERSATIONS = 50
    filtered_results = [
        r for r in results
        if 1 <= len([c for c in r["conversations"].values() if c.strip()]) <= MAX_CONVERSATIONS
    ]
    
    # 4. Write to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_results, f, indent=2, ensure_ascii=False)
        
    print(f"\nPreprocessing complete.")
    print(f"Processed {len(filtered_results)} side character personas.")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()