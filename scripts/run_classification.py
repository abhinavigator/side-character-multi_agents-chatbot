# scripts/run_classification.py

import os
import json
import sys
import logging
import time
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# --- Imports from our app modules ---
from src.side_character_app.classification.classifier import classify_character
from src.side_character_app.classification.schemas import SideCharacterClassification

# --- Imports from libraries ---
from dotenv import load_dotenv
from google import genai

def main():
    """Main function to run the classification pipeline with robust resume and retry logic."""
    # --- 1. Setup ---
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    
    client = genai.Client(api_key=api_key)

    # --- 2. Define Paths ---
    project_root = Path(__file__).resolve().parents[1]
    input_file = project_root / "data" / "processed" / "side_character_personas.json"
    output_dir = project_root / "data" / "processed"
    log_dir = project_root / "logs"

    output_jsonl = output_dir / "side_character_labeled_conversations.jsonl"
    final_json = output_dir / "side_character_labeled_conversations.json"
    log_file = log_dir / "classification_log.txt"

    output_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    # --- 3. Setup Logger ---
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("----- Started new classification session -----")

    # --- 4. Initialize Summary Stats and Resume Logic ---
    print("Initializing stats and checking for existing data to resume...")
    character_label_counts = defaultdict(int)
    conversation_label_counts = defaultdict(int)
    confidence_buckets = defaultdict(int) # **FIX**: Added back
    existing_ids = set()
    _counted_characters_from_file = set()

    if output_jsonl.exists():
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    key = (data["character_name"], data["movie_title"])
                    existing_ids.add(key)
                    
                    # Populate stats from the existing file.
                    label = data["label"]
                    confidence = data.get("confidence")
                    conversation_label_counts[label] += 1
                    
                    if key not in _counted_characters_from_file:
                        character_label_counts[label] += 1
                        # **FIX**: Populate confidence buckets from resumed data
                        if isinstance(confidence, (int, float)):
                            for threshold in range(5, 11):
                                if confidence >= threshold:
                                    confidence_buckets[f"{threshold-1}+"] += 1
                        _counted_characters_from_file.add(key)

                except (json.JSONDecodeError, KeyError):
                    logging.warning(f"Skipping malformed line during resume scan: {line.strip()}")
    
    print(f"Resuming. Found {len(existing_ids)} characters already processed.")

    # --- 5. Load Source Data ---
    with open(input_file, "r", encoding="utf-8") as f:
        character_entries = json.load(f)

    # --- 6. Main Classification Loop with Correct Retry Logic ---
    with open(output_jsonl, "a", encoding="utf-8") as f_out:
        for entry in tqdm(character_entries, desc="Classifying Characters"):
            entry_id = (entry["side_character_name"], entry["movie_title"])
            if entry_id in existing_ids:
                continue
            
            max_retries = 5
            retries = max_retries
            result = None

            while retries > 0:
                try:
                    result = classify_character(client, entry)
                    break # Success! Exit the retry loop.
                except Exception as e:
                    # **THE FIX**: Check the error message text for rate limit indicators.
                    error_text = str(e).upper()
                    if "429" in error_text and ("RESOURCE_EXHAUSTED" in error_text or "TOO MANY REQUESTS" in error_text):
                        retries -= 1
                        wait_time = 60 # Wait 60 seconds
                        logging.warning(f"RATE LIMIT HIT for {entry_id}. Retries left: {retries}. Waiting for {wait_time}s.")
                        if retries > 0:
                            time.sleep(wait_time)
                        else:
                            logging.error(f"MAX RETRIES FAILED for {entry_id} due to rate limiting. Skipping character.")
                    else:
                        # This is a different, truly non-recoverable error
                        logging.error(f"NON-RECOVERABLE ERROR for {entry_id}: {e}")
                        break # Exit retry loop for non-rate-limit errors
            
            if result is None:
                continue

            # --- Process successful result ---
            for conv_id, conv_text in entry["conversations"].items():
                output_entry = {
                    "character_name": entry["side_character_name"], "movie_title": entry["movie_title"],
                    "genre": entry.get("genre", []), "conversation_id": conv_id, "conversation": conv_text,
                    "label": result.label, "confidence": result.confidence
                }
                f_out.write(json.dumps(output_entry) + "\n")
            
            # Update stats for the newly processed character
            character_label_counts[result.label] += 1
            conversation_label_counts[result.label] += len(entry["conversations"])
            # **FIX**: Update confidence buckets for new data
            for threshold in range(5, 11):
                if result.confidence >= threshold:
                    confidence_buckets[f"{threshold}+"] += 1
            
            logging.info(f"SUCCESS: {entry_id} -> {result.label} (Confidence: {result.confidence})")

    # --- 7. Final Conversion and Summary ---
    print("\nClassification loop complete. Converting final JSONL to JSON...")
    data = [json.loads(line) for line in open(output_jsonl, "r", encoding="utf-8")]
    with open(final_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Conversion complete. Wrote {len(data)} conversations to {final_json.name}.")

    # **FIX**: Print all three summary sections correctly
    print("\n--- Final Dataset Summary (Total) ---")
    print("Characters per label:")
    for label, count in sorted(character_label_counts.items()):
        print(f"  {label}: {count}")

    print("\nConversations per label:")
    for label, count in sorted(conversation_label_counts.items()):
        print(f"  {label}: {count}")
        
    print("\nConfidence Score Distribution (Characters):")
    for k in sorted(confidence_buckets.keys()):
        print(f"  {k}: {confidence_buckets[k]} characters")

if __name__ == "__main__":
    main()