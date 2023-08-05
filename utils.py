# util.py

import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(anonymized_telemetry=False))

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch, Chroma
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader, PDFPlumberLoader, PyMuPDFLoader


import os
import shutil
import openai
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

openai.api_key = os.environ["OPENAI_API_KEY"]
llm_name = os.environ["LLM_NAME"]
user_files_directory = os.environ["USER_FILES_DIRECTORY"]

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prettify_source_documents(result):
    source_documents_printout = f"来源信息:\n\n"
    divider = "----------------------------------------\n"
    for doc in result["source_documents"]:
        source_documents_printout += f"{doc.page_content}\n"
        source_documents_printout += f"""{divider}来源文件：{doc.metadata['source']}\n"""
        source_documents_printout += (
            f"""{divider}页码：{doc.metadata['page']}/{doc.metadata['total_pages']}\n\n"""
        )
    return source_documents_printout


def prettify_chat_history(result):
    chat_history_printout = f"历史对话:\n\n"
    for chat in result["chat_history"]:
        current_role = chat.to_json()["id"][3].replace("Message", "")
        current_content = chat.to_json()["kwargs"]["content"]
        chat_history_printout += f"{current_role}: {current_content}\n"
    return chat_history_printout


def old_load_db(file, chain_type="stuff", k=2, mmr=False, chinese=True):
    # load documents
    loader = PDFPlumberLoader(
        file
    )  # replaced pypdf with pdfplumber for better Chinese support
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    if chinese:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=70)
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    if mmr:
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": k})

    # solved a bug where if set qa chain return_source_documents=True
    # and return_generated_question=True, it would crash
    # langchain.__version__ = 0.0.247
    # https://github.com/langchain-ai/langchain/issues/2256
    memory = ConversationBufferMemory(
        llm=llm_name,
        input_key="question",
        output_key="answer",
        memory_key="chat_history",
        return_messages=True,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
        verbose=True,
        memory=memory,
    )
    return qa


def load_pdf(file_list):
    loders = [PDFPlumberLoader(file) for file in file_list]
    docs = []
    for loader in loders:
        docs.extend(loader.load())
    return docs


def split_docs(docs, chinese=True):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    if chinese:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=70)
    splits = text_splitter.split_documents(docs)
    return splits


def create_vectordb(splits, persist_directory):
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=splits, embedding=embedding, persist_directory=persist_directory
    )
    return vectordb


def create_user_vectordb_with_initial_files(file_list, userid):
    persist_directory = f"{user_files_directory}/{userid}/chroma/"
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    docs = load_pdf(file_list)
    splits = split_docs(docs)
    vectordb = create_vectordb(splits, persist_directory)
    message = f"Created user vectordb with {len(file_list)} files. User ID: {userid}"
    return message, vectordb


def load_user_db(userid):
    embedding = OpenAIEmbeddings()
    persist_directory = f"{user_files_directory}/{userid}/chroma/"
    logger.info("load_user_db(userid)")
    logger.info(f"user_files_directory: {user_files_directory}")
    logger.info(f"Loading user db from {persist_directory}")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    return vectordb


def load_and_add_new_files_to_user_db(file_list, userid):
    docs = load_pdf(file_list)
    splits = split_docs(docs)
    vectordb = load_user_db(userid)
    _ = vectordb.add_documents(splits)
    return vectordb


def create_qa_chain(
    vectordb, llm_name="gpt-3.5-turbo-0613", chain_type="stuff", k=4, mmr=True
):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": k})
    if mmr:
        retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": k})

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
        verbose=True,
        memory=memory,
    )
    return qa_chain
