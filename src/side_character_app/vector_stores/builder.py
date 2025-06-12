# src/side_character_app/vector_store/builder.py

import json
import re
from pymilvus import MilvusClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def init_collection(client: MilvusClient, collection_name: str, dimension: int):
    """
    Drops and recreates a Milvus collection. This is the simple version
    that matches the original notebook's behavior.
    """
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
        print(f"Dropped existing collection: '{collection_name}'")
    
    client.create_collection(
        collection_name=collection_name,
        dimension=dimension
    )
    print(f"Created new collection: '{collection_name}'")

def prepare_data_for_collection(embedding_fn: GoogleGenerativeAIEmbeddings, json_path: str, target_label: str, min_confidence: int) -> list:
    """
    Loads, filters, and embeds data for a specific persona label,
    including the mandatory 'id' field, just like in the original notebook.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_path}' was not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{json_path}'. The file may be corrupted. Details: {e}")
        return []

    filtered = [entry for entry in data if entry.get("label") == target_label and entry.get("confidence", 0) >= min_confidence]
    if not filtered:
        print(f"No entries found for label '{target_label}' with confidence >= {min_confidence}. Skipping.")
        return []

    print(f"Embedding {len(filtered)} conversations for label '{target_label}'...")
    texts = [entry["conversation"] for entry in filtered]
    vectors = embedding_fn.embed_documents(texts)

    # **THE FIX**: This dictionary now exactly matches the structure from the
    # original working notebook, including the mandatory 'id' field.
    prepared_data = [{
        "id": i,
        "vector": vectors[i],
        "conversation": filtered[i]["conversation"],
        "character_name": filtered[i]["character_name"],
        "confidence": filtered[i]["confidence"],
        "genres": ",".join(filtered[i].get("genre", [])),
    } for i in range(len(filtered))]

    print(f"Prepared {len(prepared_data)} vector entries.")
    return prepared_data

def insert_data(client: MilvusClient, collection_name: str, data: list):
    """Inserts prepared data into the specified collection."""
    if not data:
        return
    res = client.insert(collection_name=collection_name, data=data)
    print(f"Inserted {res['insert_count']} entries into '{collection_name}'.")

def build_persona_vector_db(client: MilvusClient, embedding_fn: GoogleGenerativeAIEmbeddings, json_path: str, collection_name: str, label: str, min_confidence: int):
    """A full pipeline to initialize, prepare, and insert data for one persona."""
    # Determine embedding dimension from a test query
    dimension = len(embedding_fn.embed_query("test"))
    
    init_collection(client, collection_name, dimension)
    data_to_insert = prepare_data_for_collection(embedding_fn, json_path, target_label=label, min_confidence=min_confidence)
    insert_data(client, collection_name, data_to_insert)