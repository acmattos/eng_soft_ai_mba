# Ingestao e Busca Semantica com LangChain e Postgres

## Desafio MBA Engenharia de Software com IA - Full Cycle

Sistema de perguntas e respostas baseado em documentos PDF utilizando RAG
(Retrieval Augmented Generation) com LangChain, PGVector e OpenAI.

## Arquitetura

```
+-------------+     +------------------+     +------------------+
|   PDF       | --> |   Ingestao       | --> |   PostgreSQL     |
|   Document  |     |   (chunking +    |     |   + PGVector     |
|             |     |    embeddings)   |     |   (vetores)      |
+-------------+     +------------------+     +------------------+
                                                      |
                                                      v
+-------------+     +------------------+     +------------------+
|   Usuario   | --> |   Search         | --> |   Similarity     |
|   Pergunta  |     |   (embedding)    |     |   Search         |
+-------------+     +------------------+     +------------------+
                                                      |
                                                      v
+-------------+     +------------------+     +------------------+
|   Resposta  | <-- |   LLM (OpenAI)   | <-- |   Contexto       |
|             |     |   + Prompt       |     |   Relevante      |
+-------------+     +------------------+     +------------------+
```

## Estrutura do Projeto

```
ingestao-busca/
├── src/
│   ├── ingest.py    # Ingestao de PDFs e armazenamento de vetores
│   ├── search.py    # Busca por similaridade e construcao de prompts
│   └── chat.py      # Interface de chat com o usuario
├── .env             # Variaveis de ambiente
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Pre-requisitos

- Python 3.12+
- Docker e Docker Compose
- Chave de API da OpenAI

## Configuracao

### 1. Criar ambiente virtual

```bash
python -m venv .venv
```

### 2. Ativar ambiente virtual

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variaveis de ambiente

Edite o arquivo `.env` com suas credenciais:

```env
OPENAI_API_KEY=sua-chave-aqui
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=faturamento
PDF_PATH=./document.pdf
```

### 5. Iniciar banco de dados

```bash
docker compose up -d
```

## Execucao

### Passo 1: Ingerir documento PDF

Carrega o PDF, divide em chunks e armazena os vetores no PostgreSQL:

```bash
python src/ingest.py
```

**Saida esperada:**
```
Document ingested!
```

### Passo 2: Testar busca (opcional)

Testa a busca por similaridade e visualiza o prompt gerado:

```bash
python src/search.py
```

### Passo 3: Iniciar chat

Inicia a interface de chat interativo:

```bash
python src/chat.py
```

**Exemplo de uso:**
```
===============================================
= Para encerrar a sessao do chat, digite: SAIR
===============================================

Faca a sua pergunta: Qual o faturamento da Empresa SuperTechIABrazil?

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: R$ 10.000.000,00

Faca a sua pergunta: SAIR

Chat encerrado! Obrigado por utilizar.
```

**Exemplo de perguntas:**
```
- Qual o ano de fundação da empresa Bronze Saúde Comércio? (1959)
- Qual o faturamento da empresa Pacto Varejo EPP? (R$ 105.139,46)
- QUal o nome da empresa com faturamento de 2.090.439,49? (Quantum Consultoria EPP)
- Qual o nome da empresa com ano de fundação 1990? (Nobre Hotelaria Indústria, 
Nobre Mineração EPP, Aurora Têxtil Participações)
- Qual é o ando de fundação e o faturamento da empresa Nobre Hotelaria Indústria? 
(Ano de fundação: 1990. Faturamento: R$ 34.225.753,40.)
```

## Tecnologias Utilizadas

| Tecnologia | Descricao |
|------------|-----------|
| LangChain | Framework para aplicacoes LLM |
| OpenAI | Embeddings e modelo de chat |
| PGVector | Extensao PostgreSQL para vetores |
| PostgreSQL | Banco de dados relacional |
| Docker | Containerizacao |

## Modulos

### ingest.py
- Carrega PDFs usando `PyPDFLoader`
- Divide documentos em chunks com `RecursiveCharacterTextSplitter`
- Gera embeddings com `OpenAIEmbeddings`
- Armazena vetores no `PGVector`

### search.py
- Busca documentos similares por embedding
- Converte resultados em contexto
- Monta prompt com template estruturado

### chat.py
- Interface interativa de perguntas e respostas
- Integra busca semantica com LLM
- Loop de conversacao ate comando "SAIR"
