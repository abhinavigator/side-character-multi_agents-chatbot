# src/chosen_app/data_processing/builder.py

import pandas as pd
import re

def parse_line_ids(raw_text: str) -> list:
    """Extracts a list of line IDs (e.g., 'L1045') from the raw string."""
    return re.findall(r"L\d+", raw_text)

def build_side_character_conversations(dataframes: dict) -> list:
    """
    Constructs a dataset of conversations involving one main character and one side character.

    Args:
        dataframes: A dictionary of DataFrames from the loader module.

    Returns:
        A list of dictionaries, each representing a side character's persona.
    """
    titles_df = dataframes["titles"]
    characters_df = dataframes["characters"]
    lines_df = dataframes["lines"]
    conversations_df = dataframes["conversations"]

    # --- Build useful maps for efficient lookups ---
    credit_pos_map = dict(zip(characters_df["characterID"], characters_df["credit_pos"]))
    char_name_map = dict(zip(characters_df["characterID"], characters_df["character_name"]))
    movie_name_map = dict(zip(titles_df["movieID"], titles_df["movie_title"]))
    movie_genre_map = dict(zip(titles_df["movieID"], titles_df["genres"]))
    line_map = dict(zip(lines_df["lineID"], zip(lines_df["character_name"], lines_df["text"])))

    # --- Identify main and side characters based on credit position ---
    main_chars = {cid for cid, pos in credit_pos_map.items() if pos in [1, 2, 3]}
    side_chars = {cid for cid, pos in credit_pos_map.items() if pos >= 4}

    results = []
    print("Building structured conversations...")
    for _, row in conversations_df.iterrows():
        char1, char2, movie_id, raw_utt_ids = row["char1ID"], row["char2ID"], row["movieID"], row["utteranceIDs"]
        utt_ids = parse_line_ids(raw_utt_ids)

        # Ensure we have metadata for this movie
        if movie_id not in movie_genre_map or movie_id not in movie_name_map:
            continue

        # Check if the conversation is between a main and a side character
        if (char1 in side_chars and char2 in main_chars) or (char2 in side_chars and char1 in main_chars):
            side_id = char1 if char1 in side_chars else char2
            side_name = char_name_map.get(side_id, f"Unknown ({side_id})")
            movie_name = movie_name_map[movie_id]
            genres = movie_genre_map[movie_id]

            conv_lines = [f"{line_map[lid][0]}: {line_map[lid][1]}" for lid in utt_ids if lid in line_map]
            conv_text = "\n".join(conv_lines).strip()

            if not conv_text:
                continue

            # Aggregate conversations for the same character in the same movie
            existing_entry = next((e for e in results if e["side_character_name"] == side_name and e["movie_title"] == movie_name), None)
            if existing_entry:
                conv_key = f"conv{len(existing_entry['conversations']) + 1}"
                existing_entry["conversations"][conv_key] = conv_text
            else:
                results.append({
                    "side_character_name": side_name,
                    "movie_title": movie_name,
                    "genre": genres,
                    "conversations": {"conv1": conv_text}
                })

    return results