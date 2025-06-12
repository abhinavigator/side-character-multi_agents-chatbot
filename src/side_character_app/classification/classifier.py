# src/side_character_app/classification/classifier.py

from google import genai
from typing import Dict
import re
from .schemas import SideCharacterClassification

MODEL_NAME = "gemini-2.0-flash"

def build_prompt(entry: Dict) -> str:
    """Constructs the classification prompt for a given character entry."""
    character = entry["side_character_name"]
    movie = entry["movie_title"]
    genre = ", ".join(entry.get("genre", []))
    
    # Sort conversations by their key number to ensure chronological order
    sorted_convs = sorted(entry["conversations"].items(), key=lambda x: int(re.search(r'\d+', x[0]).group()))
    dialog_text = "\n\n".join(conv for _, conv in sorted_convs)

    prompt = f"""
You are a film analysis AI. Based on the following character dialogues, classify the character into one of four primary side-character archetypes.

Also, provide a confidence score from 1 to 10 indicating how strongly you believe the character fits that archetype.

Character: {character}
Movie: {movie}
Genres: {genre}

Dialogues:
{dialog_text}

Choose exactly one of the following labels:

- Comedic Relief: A character who provides humor and lightens the mood.
- Wise Mentor: An experienced, trusted advisor who guides the protagonist.
- Skeptical Realist: A grounded, often cynical character who questions plans and points out harsh realities.
- Loyal Sidekick: A faithful companion who offers emotional support and stands by the protagonist.

Return your output in a structured JSON format.
"""
    return prompt.strip()


# **FIX**: Reverted to the original, working API call structure.
def classify_character(client: genai.client.Client, entry: Dict) -> SideCharacterClassification:
    """
    Calls the Gemini API to classify a character based on their dialogues.

    Args:
        client: The initialized Gemini API client.
        entry: A dictionary containing the character's data.

    Returns:
        A validated SideCharacterClassification object.
    """
    prompt = build_prompt(entry)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": SideCharacterClassification,
        },
    )

    # The original .parsed attribute is correct for this client structure
    return response.parsed