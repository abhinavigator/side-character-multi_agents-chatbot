# src/side_character_app/app/state.py

from typing import TypedDict, Annotated, List, Dict
from langchain_core.messages import BaseMessage

# Define the archetypes that will be used as keys throughout the app
ARCHETYPES = ["Wise Mentor", "Comedic Relief", "Skeptical Realist", "Loyal Sidekick"]

class GraphState(TypedDict):
    """The state of our conversational graph."""
    input: str
    user_choice: str
    main_conversation: Annotated[List[BaseMessage], lambda x, y: x + y]
    private_conversations: Dict[str, List[BaseMessage]]
    next: str

def initialize_state() -> GraphState:
    """Returns a fresh, properly structured state dictionary."""
    return GraphState(
        input="",
        user_choice="",
        main_conversation=[],
        private_conversations={archetype: [] for archetype in ARCHETYPES},
        next=""
    )