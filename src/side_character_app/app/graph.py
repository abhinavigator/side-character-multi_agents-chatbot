# src/side_character_app/app/graph.py

from typing import Dict, TypedDict, List
from functools import partial
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from .state import GraphState, ARCHETYPES

def _format_conversation_history(messages: List[BaseMessage]) -> str:
    """Helper function to format the last few turns of the conversation for the router's context."""
    # Take the last 4 messages to provide recent context, or fewer if the history is short
    history = []
    for msg in messages[-4:]:
        if isinstance(msg, HumanMessage):
            history.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            # The agent's name is not stored on the AIMessage, so we'll use a generic label
            history.append(f"Agent: {msg.content}")
    
    if not history:
        return "No previous conversation history."
        
    return "\n".join(history)


def router_node(state: GraphState, llm, agents: dict) -> dict:
    """
    Decides the next agent based on conversation history and the latest user input.
    """
    # Allow user to override the router
    user_choice = state.get('user_choice')
    if user_choice and user_choice in agents:
        print(f"--- User Choice: Routing directly to {user_choice} ---")
        return {"next": user_choice}
    
    # --- Prompt Improvement ---
    # 1. Give the AI a more professional persona.
    # 2. Provide clear, detailed criteria for each choice.
    # 3. Include recent conversation history for context.
    # 4. Instruct it to reason step-by-step.
    
    conversation_history = _format_conversation_history(state["main_conversation"])
    latest_user_message = state["input"]

    system_prompt = f"""You are a master conversational director. Your job is to analyze the user's message, considering the recent conversation history, and route it to the most appropriate specialist archetype.

**ARCHETYPE ROLES:**
- **Wise Mentor:** Choose for questions about life purpose, meaning, wisdom, and abstract guidance.
- **Comedic Relief:** Choose when the user is sad, expresses a desire for humor, or the topic is very lighthearted.
- **Skeptical Realist:** Choose for requests for critical feedback, analysis of plans, risk assessment, or fact-based evaluation.
- **Loyal Sidekick:** Choose when the user is venting, expressing fear or frustration, or needs emotional support and encouragement.

**RECENT CONVERSATION HISTORY:**
{conversation_history}

**LATEST USER MESSAGE:**
"{latest_user_message}"

Based on the history and especially the LATEST USER MESSAGE, which archetype is the single best fit to respond? Think step-by-step:
1. What is the user's primary intent (e.g., seeking facts, seeking comfort, seeking a laugh)?
2. What is their emotional tone (e.g., analytical, anxious, happy)?
3. Considering these, which archetype's role is the best match?

Return your final decision in the required structured format.
"""
    
    # Define the expected structured output
    class RouteQuery(TypedDict):
        archetype: str
        
    structured_llm = llm.with_structured_output(RouteQuery, include_raw=False)
    
    print("--- Router is deliberating... ---")
    route = structured_llm.invoke(system_prompt)
    archetype = route.get('archetype')
    
    if archetype and archetype in agents:
        print(f"--- Router Decision: Route to {archetype} ---")
        return {"next": archetype}
    
    print("--- Router Fallback: Could not determine a clear route. Ending turn. ---")
    return {"next": "END"}


def agent_node(state: GraphState, agents: dict) -> dict:
    """Executes the chosen agent and correctly updates the memory channels."""
    archetype = state["next"]
    agent_executor = agents[archetype]
    user_input = state["input"]
    
    # Assemble the full conversational history for this agent
    main_history = state["main_conversation"]
    private_history = state["private_conversations"][archetype]
    chat_history = main_history + private_history
    
    # **THE FIX IS HERE**: We invoke the agent with a dictionary that matches
    # the new prompt's variables: 'input' and 'chat_history'.
    result = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    
    response_message = AIMessage(content=result["output"])
    
    # The memory update logic remains correct
    new_history = [HumanMessage(content=user_input), response_message]
    if state.get('user_choice'):
        new_private_convo = state["private_conversations"][archetype] + new_history
        return {"private_conversations": {**state["private_conversations"], archetype: new_private_convo}}
    else:
        new_private_convo = state["private_conversations"][archetype] + new_history
        return {
            "main_conversation": state["main_conversation"] + new_history,
            "private_conversations": {**state["private_conversations"], archetype: new_private_convo}
        }



def create_graph(llm, agents: dict):
    """Constructs and compiles the conversational graph."""
    graph = StateGraph(GraphState)
    
    # Use partial to inject dependencies into the node functions, keeping them clean
    bound_router_node = partial(router_node, llm=llm, agents=agents)
    bound_agent_node = partial(agent_node, agents=agents)
    
    graph.add_node("router", bound_router_node)
    for archetype in ARCHETYPES:
        graph.add_node(archetype, bound_agent_node)
    
    graph.set_entry_point("router")
    
    edge_map = {archetype: archetype for archetype in ARCHETYPES}
    edge_map["END"] = END
    graph.add_conditional_edges("router", lambda x: x["next"], edge_map)
    
    for archetype in ARCHETYPES:
        graph.add_edge(archetype, END)
    
    return graph.compile() 