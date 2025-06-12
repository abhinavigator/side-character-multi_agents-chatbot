# scripts/run_app.py

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# --- Imports from our app modules ---
from src.side_character_app.app.state import initialize_state, ARCHETYPES
from src.side_character_app.app.agents import create_all_agents
from src.side_character_app.app.graph import create_graph

# --- Imports from libraries ---
from pymilvus import MilvusClient
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

def main():
    """Main function to run the Side Character App CLI."""
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
    embedding_fn = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=google_api_key)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, temperature=0.7)

    # --- 4. Build Core App Components ---
    print("Creating agents and compiling graph...")
    agents = create_all_agents(llm, client, embedding_fn)
    app = create_graph(llm, agents)
    print("✅ Application is compiled and ready!")

    # --- 5. Main CLI Execution Loop ---
    cli_map = {"M": "Wise Mentor", "C": "Comedic Relief", "S": "Skeptical Realist", "L": "Loyal Sidekick", "N": ""}
    conversation_state = initialize_state()

    print("\n" + "="*60)
    print("     🤖 Welcome to the Side Character App! 💬")
    print("   Type 'exit' as your query to end the session.")
    print("="*60)

    while True:
        print("\n" + "-"*60)
        user_input = input('_> Your Query: ')
        if user_input.lower() == "exit":
            break

        persona_choice_key = ""
        while persona_choice_key.upper() not in cli_map:
            persona_choice_key = input("_> Choose Archetype ([M]entor, [C]omedic, [S]keptic, [L]oyal, or [N]one for Router): ")
        user_choice = cli_map[persona_choice_key.upper()]

        input_for_turn = {**conversation_state, "input": user_input, "user_choice": user_choice}
        
        print("\n----------------- App is processing... -----------------")
        final_state = app.invoke(input_for_turn, config={"configurable": {"thread_id": "main_convo"}})

        last_agent = final_state.get('next')
        if last_agent and last_agent != "END":
            response = final_state['private_conversations'][last_agent][-1].content
            print("\n---------------------- Response ----------------------")
            print(f"💬 {last_agent} Says:")
            print(response)
        else:
            print("\n💬 The app has ended the conversation or no agent was called.")
            
        conversation_state = final_state

    print("\nThank you for chatting!")

if __name__ == "__main__":
    main()