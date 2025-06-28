# {{{ imports
import pandas as pd
import numpy as np
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import AzureSearch
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

import os
import json
from dotenv import load_dotenv
load_dotenv()
# }}}

# {{{ env variables
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')  # gpt35-financial
AZURE_COGNITIVE_SEARCH_ENDPOINT = os.getenv('AZURE_COGNITIVE_SEARCH_ENDPOINT')
AZURE_COGNITIVE_SEARCH_API_KEY = os.getenv('AZURE_COGNITIVE_SEARCH_API_KEY')
AZURE_COGNITIVE_SEARCH_INDEX_NAME = os.getenv('AZURE_COGNITIVE_SEARCH_INDEX_NAME')

AZURE_OPENAI_API_VERSION = '2023-05-15'
EMBEDDING_DEPLOYMENT = 'text-embedding-ada-002'
# }}}

# {{{ load data
def load_docs(file_path):
    docs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            docs.append(Document(**data))
    return docs
# }}}

# Load documents
docs = load_docs('./bob_website_data.jsonl')

# Check doc lengths
docs_length = [len(doc.page_content) for doc in docs]
print(f"doc lengths\nmin: {min(docs_length)} \navg.: {round(np.mean(docs_length), 1)} \nmax: {max(docs_length)}")

# {{{ split into chunks
def chunk_docs(doc, chunk_size=700, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    return splitter.create_documents([doc.page_content], metadatas=[doc.metadata])

chunked_docs = [chunk_docs(doc) for doc in docs]
flattened_chunked_docs = [chunk for sublist in chunked_docs for chunk in sublist]
# }}}

# {{{ initialize Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=EMBEDDING_DEPLOYMENT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY
)
# }}}

# {{{ initialize Azure Search vector store
vector_store = AzureSearch(
    azure_search_endpoint=AZURE_COGNITIVE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_COGNITIVE_SEARCH_API_KEY,
    index_name=AZURE_COGNITIVE_SEARCH_INDEX_NAME,
    embedding_function=embeddings.embed_query
)

# Uncomment if you want to upload new documents
# vector_store.add_documents(flattened_chunked_docs)
# }}}

# {{{ initialize chat model
llm = AzureChatOpenAI(
    deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
    openai_api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.1
)
# }}}

# {{{ Prompt & Conversational QA chain
prompt_template = """
You are an expert financial advisor from Bank of Baroda, assisting users with specific Bank of Baroda products based on their financial data and user profile.
Context:
{context}
Chat history:
{chat_history}
Query: {question}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history", "question"])

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=True,
    verbose=True
)

chat_history = []
# }}}

# {{{ function: query_vector_db
def query_vector_db(query: str):
    result = qa({
        "question": query,
        "chat_history": chat_history
    })

    sources = [doc.metadata.get("source") for doc in result['source_documents']]
    return {
        "answer": result['answer'],
        "sources": sources
    }
# }}}
