from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

load_dotenv()

##--                     ENVIRONMENT VARIABLES GETTERS                      --##
def get_pdf_path() -> str:
    """
    Get the path to the PDF file from the environment variable PDF_PATH.

    Args:
        None
    Returns:
        str: The path to the PDF file.
    Raises:
        RuntimeError: If the environment variable PDF_PATH is not set.
    """
    path = os.getenv("PDF_PATH")
    if not path:
        raise RuntimeError("Environment variable PDF_PATH is not set!")    
    return path

def get_embedding_model_name() -> str:
    """
    Get the name of the embedding model from the environment variable 
    OPENAI_EMBEDDING_MODEL.

    Args:
        None
    Returns:
        str: The name of the embedding model.
    Raises:
        RuntimeError: If the environment variable OPENAI_EMBEDDING_MODEL is not 
        set.
    """
    model = os.getenv("OPENAI_EMBEDDING_MODEL")
    if not model:
        raise RuntimeError(
            "Environment variable OPENAI_EMBEDDING_MODEL is not set!"
        )        
    return model

def get_collection_name() -> str:
    """
    Get the PGVector's collection name from the environment variable 
    PG_VECTOR_COLLECTION_NAME.

    Args:
        None
    Returns:
        str: The collection name.
    Raises:
        RuntimeError: If the environment variable PG_VECTOR_COLLECTION_NAME is
                      not set.
    """
    collection = os.getenv("PG_VECTOR_COLLECTION_NAME")
    if not collection:
        raise RuntimeError("Environment variable PG_VECTOR_COLLECTION_NAME is not set!")
    return collection

def get_db_connection() -> str:
    """
    Get the connection string from the environment variable DATABASE_URL.

    Args:
        None
    Returns:
        str: The connection string.
    Raises:
        RuntimeError: If the environment variable DATABASE_URL is not set.
    """
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("Environment variable DATABASE_URL is not set!")
    return url
##--                     ENVIRONMENT VARIABLES GETTERS                      --##

##--                       DOCUMENT PROCESSING                              --##
def get_documents(file_path: str) -> list[Document]:
    """
    Get the documents from the file path.

    Args:
        file_path: The path to the PDF file to be loaded.
    Returns:
        list[Document]: List of documents, where each document contains a piece 
                        of text and associated metadata.
    """
    return PyPDFLoader(file_path).load()

def get_split_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        documents: List of documents to be split.
    Returns:
        list[Document]: List of document chunks with configured size (1000)
                        and overlap (150).
    """    
    return RecursiveCharacterTextSplitter(
        chunk_size      = 1000,
        chunk_overlap   = 150,
        add_start_index = False,
    ).split_documents(documents)


def get_normalized_documents(split_documents: list[Document]) \
    -> tuple[list[Document], list[str]]:
    """
    Get normalized documents by removing empty metadata and generate unique IDs.

    Args:
        split_documents: List of document chunks to be processed.
    Returns:
        tuple[list[Document], list[str]]: Documents with cleaned metadata and
        their corresponding IDs (doc-0, doc-1, ...).
    """
    normalized_documents = [
        Document(
            page_content = document.page_content,
            metadata     = {
                k: v for k, v in document.metadata.items() \
                if v not in ("", None)
            },
        ) for document in split_documents
    ]
    indexes = [f"doc-{i}" for i in range(len(normalized_documents))]

    return normalized_documents, indexes
##--                       DOCUMENT PROCESSING                              --##

##--                         VECTOR STORE                                   --##
def get_embeddings(model_name: str) -> Embeddings:
    """
    Get the new created OpenAI embeddings model instance.

    Args:
        model_name: Name of the OpenAI embedding model (e.g., text-embedding-3-small).
    Returns:
        OpenAIEmbeddings: Configured embeddings model instance.
    """    
    return OpenAIEmbeddings(
        model = model_name
    )

def get_vector_store(
        embeddings: Embeddings,
        connection: str,
        collection_name: str
    ) -> PGVector:
    """
    Get the new created a PGVector store instance for storing and retrieving 
    vectors.

    Args:
        embeddings: The embeddings model (LangChain Embeddings).
        connection: PostgreSQL connection string.
        collection_name: Name of the collection to store vectors.
    Returns:
        PGVector: Configured vector store instance.
    """
    return PGVector(
        embeddings      = embeddings,
        connection      = connection,
        collection_name = collection_name,
        use_jsonb       = True,
    )
##--                           VECTOR STORE                                 --##

##--                            ORCHESTRATION                               --##
def ingest_pdf() -> None:
    """
    Orchestrate PDF ingestion: load, split, enrich and store in vector database.

    Processing steps:
    1. Load PDF from path defined in PDF_PATH environment variable.
    2. Split documents into chunks with overlap for better retrieval.
    3. Normalize metadata and generate unique IDs.
    4. Store documents with embeddings in PGVector.
    """    
    split_documents = get_split_documents(
        get_documents(
            get_pdf_path()
        )
    )

    normalized_documents, indexes = get_normalized_documents(split_documents)
    
    get_vector_store(
        embeddings      = get_embeddings(get_embedding_model_name()),
        connection      = get_db_connection(),
        collection_name = get_collection_name()
    ).add_documents(
        documents = normalized_documents, 
        ids       = indexes
    )
##--                            ORCHESTRATION                               --##

##--                                  MAIN                                  --##
if __name__ == "__main__":
    ingest_pdf()
    print("Document ingested!")
