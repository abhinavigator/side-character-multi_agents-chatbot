# src/chosen_app/data_processing/loader.py

import pandas as pd
import re
import os

def parse_genres(raw_text: str) -> list:
    """Extracts a list of genres from the raw string representation."""
    return re.findall(r"'([^']+)'", raw_text)

def load_and_clean_data(data_dir: str) -> dict:
    """
    Loads and performs initial cleaning on the four Cornell Movie-Dialogs corpus files.

    Args:
        data_dir: The path to the directory containing the raw .tsv files.

    Returns:
        A dictionary of cleaned pandas DataFrames.
    """
    # --- Load titles ---
    titles_path = os.path.join(data_dir, "movie_titles_metadata.tsv")
    titles_df = pd.read_csv(titles_path, sep="\t", header=None,
                            names=["movieID", "movie_title", "movie_year", "rating", "votes", "genres"])
    titles_df["genres"] = titles_df["genres"].apply(lambda x: parse_genres(x) if isinstance(x, str) else [])

    # --- Load characters ---
    characters_path = os.path.join(data_dir, "movie_characters_metadata.tsv")
    characters_df = pd.read_csv(characters_path, sep="\t", header=None,
                                names=["characterID", "character_name", "movieID", "movie_title", "gender", "credit_pos"],
                                engine="python", on_bad_lines="skip")
    # Clean credit position
    characters_df = characters_df[characters_df["credit_pos"] != "?"]
    characters_df = characters_df.dropna(subset=["credit_pos"])
    characters_df["credit_pos"] = characters_df["credit_pos"].astype(int)

    # --- Load lines ---
    lines_path = os.path.join(data_dir, "movie_lines.tsv")
    parsed_rows = []
    with open(lines_path, "r", encoding="iso-8859-1") as f: # Use iso-8859-1 for better compatibility
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < 5:
                continue
            parsed_rows.append({
                "lineID": fields[0],
                "characterID": fields[1],
                "movieID": fields[2],
                "character_name": fields[3],
                "text": "\t".join(fields[4:]).replace('""', '"')
            })
    lines_df = pd.DataFrame(parsed_rows)

    # --- Load conversations ---
    conversations_path = os.path.join(data_dir, "movie_conversations.tsv")
    conversations_df = pd.read_csv(conversations_path, sep="\t", header=None,
                                   names=["char1ID", "char2ID", "movieID", "utteranceIDs"],
                                   engine="python", on_bad_lines="skip")

    print("Data loaded and cleaned successfully.")
    return {
        "titles": titles_df,
        "characters": characters_df,
        "lines": lines_df,
        "conversations": conversations_df
    }