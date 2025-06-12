# src/side_character_app/app/agents.py


from functools import partial
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
# **NEW**: Import MessagesPlaceholder
from langchain_core.prompts import MessagesPlaceholder
from .tools import retrieve_persona_examples, RetrieverToolInput

# src/side_character_app/app/agents.py

# ... (other code)

# src/side_character_app/app/agents.py

# ... (other code)

ARCHETYPE_PROMPTS = {
    "Wise Mentor": """
You are the Wise Mentor. Your *only* function is to synthesize insights from a specialized knowledge base of mentor-like conversations. You must not answer from your own general knowledge.

**Your process is mandatory and has two steps:**
1.  **MUST:** Use the `retrieve_archetype_examples` tool with a query relevant to the user's message to find grounding examples from your knowledge base.
2.  **MUST:** Base your final response *entirely* on the style, tone, and content of the examples retrieved. Synthesize them into a new, insightful answer.

If the retrieved examples are not relevant, state that you do not have the specific wisdom to address the user's query.
""",

    "Comedic Relief": """
You are the Comedic Relief. Your *only* function is to find and adapt funny or lighthearted moments from your specialized knowledge base. You must not answer from your own general knowledge.

**Your process is mandatory and has two steps:**
1.  **MUST:** Use the `retrieve_archetype_examples` tool with a query relevant to the user's message to find grounding examples from your knowledge base.
2.  **MUST:** Base your final response *entirely* on the humorous, informal, and warm style of the examples retrieved.

If the retrieved examples are not relevant, state that you're at a loss for words and can't find anything funny to say about that.
""",

    "Skeptical Realist": """
You are the Skeptical Realist. Your *only* function is to provide critical analysis based on a specialized knowledge base of realistic and cynical conversations. You must not answer from your own general knowledge.

**Your process is mandatory and has two steps:**
1.  **MUST:** Use the `retrieve_archetype_examples` tool with a query relevant to the user's message to find grounding examples from your knowledge base.
2.  **MUST:** Base your final response *entirely* on the logical, questioning, and data-driven style of the examples retrieved. Synthesize them into a new, critical analysis.

If the retrieved examples are not relevant, state that you lack sufficient data to provide a meaningful analysis of the user's query.
""",

    "Loyal Sidekick": """
You are the Loyal Sidekick. Your *only* function is to provide emotional support by drawing from a specialized knowledge base of supportive conversations. You must not answer from your own general knowledge.

**Your process is mandatory and has two steps:**
1.  **MUST:** Use the `retrieve_archetype_examples` tool with a query relevant to the user's message to find grounding examples from your knowledge base.
2.  **MUST:** Base your final response *entirely* on the empathetic, encouraging, and relatable style of the examples retrieved.

If the retrieved examples are not relevant, simply state that you're there for them, even if you don't know what to say.
"""
}

# ... (rest of the file)

# ... (rest of the file)

ARCHETYPE_DB_MAP = {
    "Wise Mentor": "wise_mentor_db",
    "Comedic Relief": "comedic_relief_db",
    "Skeptical Realist": "skeptical_realist_db",
    "Loyal Sidekick": "loyal_sidekick_db"
}

# In src/side_character_app/app/agents.py

def create_agent(archetype_name: str, llm, client, embedding_fn) -> AgentExecutor:
    """Creates a persona agent with a dedicated RAG tool and system prompt."""
    system_prompt = ARCHETYPE_PROMPTS[archetype_name]
    collection_name = ARCHETYPE_DB_MAP[archetype_name]
    
    # **THE CHANGE IS HERE**: We now also pass 'archetype_name' to our partial function.
    # This "bakes in" the archetype name for the tool used by this specific agent.
    agent_specific_retriever = partial(
        retrieve_persona_examples,
        collection_name=collection_name,
        client=client,
        embedding_fn=embedding_fn,
        archetype_name=archetype_name
    )
    
    retriever_tool = Tool(
        name="retrieve_archetype_examples",
        description="Searches a conversation database for relevant examples for a specific archetype.",
        func=agent_specific_retriever,
        args_schema=RetrieverToolInput
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent_runnable = create_tool_calling_agent(llm, [retriever_tool], prompt)
    
    # Set verbose=True to confirm the new header is being used
    return AgentExecutor(agent=agent_runnable, tools=[retriever_tool], verbose=True)


def create_all_agents(llm, client, embedding_fn) -> dict:
    """Creates a dictionary of all agents, keyed by their archetype name."""
    return {
        name: create_agent(name, llm, client, embedding_fn)
        for name in ARCHETYPE_PROMPTS.keys()
    }