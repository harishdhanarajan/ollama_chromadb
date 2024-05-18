from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embedding_function():
    # Embedding dimension should match collection dimensionality 768
    # embedding_dimension = 768
    embeddings = OllamaEmbeddings(model = "nomic-embed-text")
    return embeddings
