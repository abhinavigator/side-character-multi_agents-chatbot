# Side Character AI: A Multi-Agent Conversational System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-blue.svg)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0-orange.svg)](https://langchain-ai.github.io/langgraph/)

This project is a full-stack, multi-agent conversational AI system built for the Vectorial AI take-home assignment. It leverages LangGraph to orchestrate a team of specialized AI agents, each grounded in a distinct knowledge base derived from the Cornell Movie-Dialogs Corpus. The user can interact with the system through a unified Streamlit chat interface, either letting an AI router delegate tasks or by directly addressing a specific agent persona.



## ✨ Key Features

* **Multi-Agent System:** Features four distinct AI agents (Wise Mentor, Comedic Relief, Skeptical Realist, Loyal Sidekick) orchestrated by LangGraph.
* **Retrieval-Augmented Generation (RAG):** Each agent is grounded in its own partitioned knowledge base, stored in a Milvus vector database, ensuring responses are stylistically consistent and within defined boundaries.
* **Advanced Memory Management:** Implements a sophisticated memory system with a shared "public" conversation history and private, per-agent memory channels.
* **Intelligent Routing:** An AI router analyzes user intent to delegate tasks to the most appropriate agent, but also allows for direct user override.
* **Interactive UI:** A polished and responsive chat interface built with Streamlit, providing a unified view of the multi-agent conversation.
* **Modular & Reproducible Pipeline:** The entire data pipeline, from raw data processing to vector store creation, is built with modular, runnable Python scripts.

## 🏛️ Architecture & Design Decisions

This project is structured as a modular Python application to ensure a clear separation of concerns, maintainability, and reproducibility.

### 1. Project Structure

The project is organized into three main directories:

* `data/`: Contains the `raw/` dataset, `processed/` JSON files generated by the pipeline, and `vector_stores/` which holds the Milvus database.
* `src/side_character_app/`: The core source code for the application, divided into logical modules:

  * `data_processing`: Scripts to parse and structure the raw Cornell Movie-Dialogs Corpus.
  * `classification`: Scripts that use the Gemini API to label conversations into archetypes.
  * `vector_store`: Scripts to build the persona-specific vector databases.
  * `app`: The main LangGraph application logic, including state, tools, agent definitions, and graph construction.
* `scripts/`: Standalone, runnable Python scripts that execute the different stages of the pipeline (e.g., `run_preprocessing.py`, `run_app.py`).

### 2. The Data Pipeline

The journey from raw movie scripts to an intelligent agent system is handled by three sequential scripts:

1. **`scripts/run_preprocessing.py`**:
   Reads the raw `.tsv` files from `data/raw`, parses the complex relationships between movies, characters, and lines, and identifies all conversations between a "main character" (credit position 1-3) and a "side character" (credit position 4+). Outputs a structured `side_character_personas.json` file.

2. **`scripts/run_classification.py`**:
   Ingests the processed JSON file, sends each side character's complete dialogue set to the Gemini API, and asks it to classify the character into one of four archetypes (Wise Mentor, etc.) with a confidence score. Features robust retry logic for API rate limits and is resumable. Outputs `side_character_labeled_conversations.json`.

3. **`scripts/build_vector_stores.py`**:
   Partitions the labeled conversations by archetype. For each archetype, creates a dedicated collection in a Milvus vector database, embedding the conversations to enable semantic search for the RAG system.

### 3. Agent Architecture: An "Agentic" Approach

A key design decision was how to implement the persona "agents." The project uses LangChain's **`AgentExecutor`**, which provides agents with a set of tools and the autonomy to decide *if* and *when* to use them.

* **Justification:** Models a more "true" agent. For instance, the `Loyal Sidekick` can answer empathetic queries directly without retrieval, but uses its RAG tool when context-specific examples are needed.
* **Trade-off:** Tool use is not guaranteed every turn, favoring natural interaction. Agents use retrieval only when necessary, aligning with intelligent collaboration.

### 4. Memory Management: Shared vs. Private History

To enable sophisticated, multi-turn conversations, the system uses a hybrid memory model managed by the `GraphState`:

* **`main_conversation` (Public):** Shared transcript when using Auto Mode.
* **`private_conversations` (Private):** Per-agent logs for one-on-one interactions.
* **Agent Context:** Agents combine `main_conversation` with their private history to maintain awareness of public events and personal interactions.

## ⚙️ Setup and Installation

Follow these steps to set up the local environment and run the project.

1. **Clone the Repository**

   ```bash
   git clone <your-repo-url>
   cd side_character_project
   ```

2. **Create and Activate virtual python Environment**
   This project requires Python 3.10+.

   ```bash
   python3 -m venv venv {following this command you can create a virtual environment}
   source venv/bin/activate  {activate the virtual environment}

   ```

3. **Set Up Environment Variables**
   Create a `.env` file in the root and add your Gemini API key:

   ```
   # .env
   GEMINI_API_KEY="your_google_api_key_here"
   ```

4. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the Application

The application is run in two stages: a one-time data pipeline and the interactive chat application.

### Stage 1: Run the Data Pipeline

```bash
# 1. Preprocess the Raw Data
python scripts/run_preprocessing.py

# 2. Classify the Conversations
python scripts/run_classification.py

# 3. Build the Vector Databases
python scripts/build_vector_stores.py
```

### Stage 2: Run the Interactive Streamlit App

```bash
streamlit run app_ui.py
```

## 📖 Usage Guide

1. **Choose an Interaction Mode:**

   * **Auto Mode:** AI Router delegates your message to the best agent.
   * **Specific Archetype:** Direct your message (e.g., @Wise Mentor) to bypass the router.

2. **Chat:** Type your message and press Enter.

### Example Workflow: Planning a Party

1. **User (Auto Mode):**
   “I have to plan a party but I’m stressed. Don’t know where to start.”
   *Router → Loyal Sidekick*
   **Loyal Sidekick:** “Hey, don’t worry! Let’s make planning fun. What’s stressing you most?”

2. **User (@Skeptical Realist):**
   “What goes wrong when planning a party?”
   **Skeptical Realist:**
   “Common pitfalls:

   1. Unrealistic budget
   2. Inadequate venue
   3. Cleanup oversight
      Have you budgeted for these?”

3. **User (@Comedic Relief):**
   “Tell me a funny party story.”
   **Comedic Relief:**
   “Once I made a seven-layer dip with chocolate pudding instead of beans—it was… memorable!”

4. **User (@Wise Mentor):**
   “What’s the true purpose of a party?”
   **Wise Mentor:**
   “A party is a vessel for shared joy, reminding us that our connections matter most.”

## 🧪 Testing

Run the retriever sanity check to ensure each vector DB returns relevant context:

```bash
python scripts/test_retriever.py
```

Unit tests for deterministic helpers are ideal for future coverage.

## Limitations & Future Work

* **Memory Reasoning:** Enhance proactive summarization for long-context handling.
* **UI State Persistence:** Persist session history beyond browser refresh (e.g., database or localStorage).
* **Complex Workflows:** Enable agent-to-agent calls or subgraphs for advanced coordination.

Future improvements could include performance optimizations, additional personas, and richer UI/UX features.




