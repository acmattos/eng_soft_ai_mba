from dotenv import load_dotenv
from langchain_core.documents import Document
from typing import List, Tuple

from ingest import get_collection_name, get_db_connection, \
                   get_embedding_model_name, get_embeddings, get_vector_store

load_dotenv()

##--                                CONSTANTS                               --##
PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""
##--                                CONSTANTS                               --##

##--                                 SEARCH                                 --##
def similarity_search_with_score(question: str) \
    -> List[Tuple[Document, float]]:
    """
    Search for similar documents in PGVector based on the question.

    Args:
        question: User's question to search for similar documents.
    Returns:
        List[Tuple[Document, float]]: List of tuples containing the document
        and its similarity score (top 10 results).
    """
    store = get_vector_store(
        embeddings      = get_embeddings(get_embedding_model_name()),
        connection      = get_db_connection(),
        collection_name = get_collection_name()
    )

    return store.similarity_search_with_score(
        query = question, 
        k     = 10,
    )

def search_result_to_context(
    search_result: List[Tuple[Document, float]]
) -> str:
    """
    Convert search results to a single context string.

    Args:
        search_result: List of tuples (Document, score) from similarity search.
    Returns:
        str: Concatenated page contents separated by newlines.
    """
    return "\n".join([doc.page_content for doc, _ in search_result])
    
def search_prompt(question: str, context: str) -> str:
    """
    Build the final prompt by combining context and question.

    Args:
        question: User's question to be answered.
        context: Retrieved context from similarity search.
    Returns:
        str: Formatted prompt ready to be sent to the LLM.
    """
    return PROMPT_TEMPLATE.format(
        contexto = context,
        pergunta = question,
    )
##--                                 SEARCH                                 --##

##--                                  MAIN                                  --##
if __name__ == "__main__":
    question = "Qual o faturamento da Empresa SuperTechIABrazil?"
    question = "Qual empresa tem o faturamento de 2.090.439,49?"
    search_result = similarity_search_with_score(question)
    context = search_result_to_context(search_result)
    prompt = search_prompt(context, question)
    print("Prompt Sample\n\n")
    print(prompt)    
