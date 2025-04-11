import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class VectorDBQuerier:
    """A class to handle querying a ChromaDB vector database."""
    
    def __init__(self, collection, model):
        """
        Initialize the querier with a collection and embedding model.
        
        Args:
            collection: The ChromaDB collection to query
            model: The embedding model to encode queries
        """
        self.collection = collection
        self.model = model
    
    def query(self, query_text, n_results=3, metadata_filter=None):
        """
        Run a query against the vector database.
        
        Args:
            query_text (str): The query text
            n_results (int): Number of results to return
            include_distances (bool): Whether to include distance/similarity scores
            metadata_filter (dict): Optional filter for metadata fields
            
        Returns:
            dict: Dictionary containing formatted results
        """
        # Encode query with embedding model
        query_embedding = self.model.encode(query_text).tolist()
        
        # Prepare query parameters
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        
        # Add metadata filter if provided
        if metadata_filter:
            query_params["where"] = metadata_filter
        
        # Run the query
        search_results = self.collection.query(**query_params)
        
        # Format and display results
        formatted_results = []
        
        if len(search_results["ids"][0]) == 0:
            print("No results found.")
            return {"query": query_text, "results": []}
        
        for i, (id, document, metadata) in enumerate(zip(
                search_results["ids"][0],
                search_results["documents"][0],
                search_results["metadatas"][0]
            )):
            result = {
                "document_id": metadata['doc_id'],
                "chunk_id": metadata['chunk_id'],
                "metadata": metadata,
                "text": document
            }
            
            # Add similarity score if available
            if "distances" in search_results:
                distance = search_results["distances"][0][i]
                similarity_score = 1 - distance  # For cosine distance
                result["similarity"] = similarity_score
            
            formatted_results.append(result)
            
            # Print result
            print(f"Result {i+1}:")
            print(f"  Document: {metadata['doc_id']}")
            print(f"  Chunk: {metadata['chunk_id']}")
            
            if "distances" in search_results:
                print(f"  Similarity Score: {similarity_score:.4f}")
            
            print(f"  Text: {document[:150]}...")
            print()
        
        return {
            "query": query_text,
            "results": formatted_results,
            "raw_results": search_results
        }
    
    def display_result(self, result, text_length=150):
        """
        Display a single result entry in a formatted way.
        
        Args:
            result (dict): A single result entry
            text_length (int): Length of text preview to display
        """
        print(f"Document: {result['document_id']}")
        print(f"Chunk: {result['chunk_id']}")
        
        if 'similarity' in result:
            print(f"Similarity Score: {result['similarity']:.4f}")
        
        print(f"Text: {result['text'][:text_length]}...")
        print()
        
def get_vectordb_bocyl():
    """
    Connect to an existing ChromaDB collection and initialize the vector store.

    Args:
        collection_name (str): The name of the ChromaDB collection.
        model (str): The model to be used for embeddings.

    Returns:
        Chroma: The initialized vector store.
    """
    # 1. Connect to your existing ChromaDB collection
    db_path = "/workspace/data/vectordb/chromadb"
    collection_name = "bocyl"
    model = "BAAI/bge-small-en-v1.5"

    # Initialize the embeddings using the HuggingFace wrapper
    embedding_function = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )

    # Create the vector store using LangChain's Chroma wrapper
    # This connects to your existing collection
    vectorstore = Chroma(
        persist_directory=db_path,
        collection_name=collection_name,
        embedding_function=embedding_function
    )
    
    return vectorstore