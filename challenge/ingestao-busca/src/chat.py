from langchain_openai import ChatOpenAI
from search import search_prompt, search_result_to_context, \
                   similarity_search_with_score

import os

##--                      ENVIRONMENT VARIABLE GETTER                       --##
def get_chat_model_name() -> str:
    """
    Get the chat model name from the environment variable OPENAI_CHAT_MODEL.

    Args:
        None
    Returns:
        str: The name of the OpenAI chat model.
    Raises:
        RuntimeError: If the environment variable OPENAI_CHAT_MODEL is not set.
    """
    model = os.getenv("OPENAI_CHAT_MODEL")
    if not model:
        raise RuntimeError("Environment variable OPENAI_CHAT_MODEL is not set!")
    return model
##--                      ENVIRONMENT VARIABLE GETTER                       --##

def main() -> None:
    os.system('cls' if os.name == 'nt' else 'clear')
    chat = ChatOpenAI(model=get_chat_model_name())
    print("===============================================")
    print("= Para encerrar a sessão do chat, digite: SAIR")
    print("===============================================\n\n")
    while True:
        question = input("Faça a sua pergunta: ")
        if question.upper() == "SAIR":
            print("\n\nChat encerrado. Obrigado por utilizar!\n\n")
            break
        search_result = similarity_search_with_score(question)
        context = search_result_to_context(search_result)
        prompt = search_prompt(context, question)
        response = chat.invoke(prompt)
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"\nPERGUNTA: {question}")
        print(f"RESPOSTA: {response.content}\n")

##--                                  MAIN                                  --##
if __name__ == "__main__":
    main()
