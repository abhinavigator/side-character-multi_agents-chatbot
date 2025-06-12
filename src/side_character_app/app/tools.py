# src/side_character_app/app/tools.py

from pydantic import BaseModel, Field
from pymilvus import MilvusClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class RetrieverToolInput(BaseModel):
    """Input schema for the retriever tool."""
    query: str = Field(description="The user's query to search for relevant conversation examples.")

def format_retrieved_docs(docs: list, archetype_name: str) -> str:
    """Helper function to format Milvus search results into a clean string."""
    if not docs or not docs[0]:
        return "No relevant conversation examples were found."
    output = ""
    for i, doc in enumerate(docs[0]):
        entity = doc.get('entity', {})
        character = entity.get('character_name', 'Unknown Character')
        genres = entity.get('genres', 'unknown genre')
        conversation = entity.get('conversation', 'N/A')
        
        # **THE CHANGE IS HERE**: The header now includes the archetype.
        header = (
            f"Example {i+1}: In the following conversation, the character '{character}' "
            f"acts as a '{archetype_name}' in a movie with genres: {genres}.\n"
        )
        
        formatted_convo = f"Conversation:\n---\n{conversation}\n---\n\n"
        output += header + formatted_convo
    return output

def retrieve_persona_examples(query: str, collection_name: str, client: MilvusClient, embedding_fn: GoogleGenerativeAIEmbeddings, archetype_name: str) -> str:
    """Searches a specific persona's conversation database for relevant examples."""
    try:
        query_vector = embedding_fn.embed_query(query)
        search_res = client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=5, # Using 5 to provide more context
            output_fields=["conversation", "character_name", "genres"]
        )
        # Pass the archetype_name down to the formatting function
        return format_retrieved_docs(search_res, archetype_name=archetype_name)
    except Exception as e:
        return f"Could not retrieve examples from collection '{collection_name}' due to an error: {e}"