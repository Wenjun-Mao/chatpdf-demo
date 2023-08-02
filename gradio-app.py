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


import gradio as gr
import os
import shutil


llm_name = "gpt-3.5-turbo-0613"

def load_db(file, chain_type='stuff', k=2, mmr=False, chinese = True):
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


qa = None
process_status = False
USERID = ""

def save_file(file, userid):
    if not os.path.exists(f"user_files/{userid}/docs"):
        os.makedirs(f"user_files/{userid}/docs")

    base_file_name = os.path.basename(file.name)
    saved_file_path = os.path.join(f"user_files/{userid}/docs", base_file_name)
    shutil.move(file.name, saved_file_path)

    return saved_file_path

def save_file_and_load_db(files, userid):
    global qa
    file_paths = []
    for file in files:
        file_path = save_file(file, userid)
        file_paths.append(file_path)
    # qa = load_db(file_path)
    qa = 1
    return qa


def clear_all_files_only():
    global qa
    global process_status
    qa = None
    process_status = False
    if os.path.exists('doc'):
        for filename in os.listdir('doc'):
            file_path = os.path.join('doc', filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    return None, None, None


def delete_user(userid):
    if userid != "":
        global qa
        global process_status
        qa = None
        process_status = False
        if os.path.exists(f'user_files/{userid}'):
            shutil.rmtree(f'user_files/{userid}')
        return None, None, None, None


def process_file(files, userid):
    global process_status
    global qa
    global USERID

    USERID = userid

    if (files is not None) and (userid != ""):
        try:
            qa = save_file_and_load_db(files, userid)
        except:
            return "Error processing file."
        process_status = True
        return "文件处理完成 Processing complete."
    else:
        return "没有文件或用户ID File not uploaded."

def get_answer(question):
    global qa
    global process_status
    if process_status:
        result = qa({"question": question})
        return result["answer"], prettify_chat_history(result), prettify_source_documents(result), result['generated_question']
    else:
        error_msg = "请先上传并分析文件 Please upload and process a file first."
        return error_msg, error_msg, error_msg, error_msg


with gr.Blocks() as demo:
    gr.Markdown("# AI 论文小助手")
    userid = gr.Textbox(label="用户ID")
    pdf_upload = gr.Files(label="上传PDF文件")
    btn_process = gr.Button("分析文件")
    process_message = gr.Markdown()
    with gr.Row():
        with gr.Column(scale=4):
            current_question = gr.Textbox(label="问题")
        with gr.Column(scale=1, min_width=50):
            btn_ask = gr.Button("提问", scale=1)

    generated_question = gr.Textbox(label="根据上下文生成的问题 --（内部功能）")
    current_answer = gr.Textbox(label="回答")

    with gr.Row():
        chat_hitsory = gr.Textbox(label="历史对话", lines=10)
        source_documents = gr.Textbox(label="来源信息", lines=10)

    btn_clear = gr.Button("清除对话记录")

    btn_delete_user = gr.Button("删除用户")

    # btn_process.click(fn=save_file_and_load_db, inputs = [pdf_upload], outputs = [output_question, output_answer])
    btn_process.click(fn=process_file, inputs = [pdf_upload, userid], outputs = [process_message])
    btn_ask.click(fn=get_answer, inputs = [current_question], outputs = [current_answer, chat_hitsory, source_documents, generated_question])
    btn_clear.click(fn=clear_all_files_only, outputs = [pdf_upload, chat_hitsory, source_documents])
    btn_delete_user.click(fn=delete_user, inputs = [userid], outputs = [pdf_upload, chat_hitsory, source_documents, generated_question])

gr.close_all()
demo.launch(share=False, server_port=7878, allowed_paths=["D:\MyDocuments\03-PythonProjects\ChatGPT_Projects\chatpdf-demo\doc"])