# src/side_character_app/classification/schemas.py

from pydantic import BaseModel, Field
from typing import Literal

class SideCharacterClassification(BaseModel):
    """
    Pydantic schema for the structured output from the Gemini classification model.
    """
    label: Literal[
        "Comedic Relief",
        "Wise Mentor",
        "Skeptical Realist",
        "Loyal Sidekick"
    ] = Field(description="The single most fitting character archetype.")
    
    confidence: int = Field(
        description="A score from 1 (very unsure) to 10 (very confident) on the classification.",
        ge=1, # ge = greater than or equal to
        le=10  # le = less than or equal to
    )