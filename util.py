from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader, PDFPlumberLoader, PyMuPDFLoader

from langchain.memory import ConversationBufferMemory

import os
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key  = os.environ['OPENAI_API_KEY']


def prettify_source_documents(result):
    source_documents_printout = f'来源信息:\n\n'
    divider = '----------------------------------------\n'
    for doc in result['source_documents']:
        source_documents_printout += f'{doc.page_content}\n'
        source_documents_printout += f"""文件：{doc.metadata['source']}  """
        source_documents_printout += f"""{divider}页码：{doc.metadata['page']}/{doc.metadata['total_pages']}\n\n"""
    return source_documents_printout


def prettify_chat_history(result):
    chat_history_printout = f'历史对话:\n\n'
    for chat in result['chat_history']:
        current_role = chat.to_json()['id'][3].replace('Message', '')
        current_content = chat.to_json()['kwargs']['content']
        chat_history_printout += f'{current_role}: {current_content}\n'
    return chat_history_printout



def load_db(file, chain_type='stuff', k=2, mmr=False, chinese = True):
    llm_name = "gpt-3.5-turbo-0613"
    # load documents
    loader = PDFPlumberLoader(file)   # replaced pypdf with pdfplumber for better Chinese support
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
        input_key='question',
        output_key='answer',
        memory_key="chat_history",
        return_messages=True
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
        verbose=True,
        memory=memory
    )
    return qa