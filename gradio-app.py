import gradio as gr
import os
import shutil
import glob

from typing import IO

from util import (
    prettify_source_documents,
    prettify_chat_history,
    delete_user,
    clear_all_files_only,
)


qa = None
process_status = False
USERID = ""

# ---------- langchain Start----------
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PDFPlumberLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

embedding = OpenAIEmbeddings()

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
    vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
    vectordb.persist()
    return vectordb


def create_user_vectordb_with_initial_files(file_list, userid):
    persist_directory = f'user_files/{userid}/chroma/'
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    docs = load_pdf(file_list)
    splits = split_docs(docs)
    vectordb = create_vectordb(splits, persist_directory)
    message = f"Created user vectordb with {len(file_list)} files. User ID: {userid}"
    return message, vectordb


def load_user_db(userid):
    persist_directory = f'user_files/{userid}/chroma/'
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    return vectordb


def load_and_add_new_files_to_user_db(file_list, userid):
    persist_directory = f'user_files/{userid}/chroma/'
    docs = load_pdf(file_list)
    splits = split_docs(docs)
    vectordb = load_user_db(persist_directory)
    _ = vectordb.add_documents(splits)
    return vectordb


def create_qa_chain(vectordb, llm_name="gpt-3.5-turbo-0613", chain_type='stuff', k=4, mmr=True):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key='question',
        output_key='answer',
        return_messages=True
    )
    retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": k})
    if mmr:
        retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": k})

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


# ---------- langchain End----------

def get_user_existing_files_list(userid):
    # Check if user exists
    if os.path.exists(f"user_files/{userid}"):
        # If user exists, read existing file names
        try:
            with open(f"user_files/{userid}/file_list.txt", 'r', encoding='utf-8') as f:
                existing_files = f.read().splitlines()
        except FileNotFoundError:
            # Check if there are files in the "docs" folder
            docs_dir = f"user_files/{userid}/docs"
            if os.path.exists(docs_dir) and glob.glob(docs_dir + '/*.pdf'):
                print("User list corrupted, reloading existing files...")
                existing_files = os.listdir(docs_dir)
                
                # Delete existing db
                if os.path.exists(f'user_files/{userid}/chroma/'):
                    shutil.rmtree(f'user_files/{userid}/chroma/')
                _ = create_user_vectordb_with_initial_files(existing_files, userid)

                # Create file list
                with open(f"user_files/{userid}/file_list.txt", 'w', encoding='utf-8') as f:
                    f.write('\n'.join(existing_files))
            else:
                existing_files = []
    else:
        existing_files = []

    return existing_files


def save_file(file: IO, userid: str, existing_files: list[str]) -> tuple[str, str]:
    # if file already exists
    # return (None, "file already exists")
    # if file does not exist
    # save file and 
    # return (file_path, None)
    if not os.path.exists(f"user_files/{userid}/docs"):
        os.makedirs(f"user_files/{userid}/docs")

    base_file_name = os.path.basename(file.name)

    # Check if file already exists in the list
    if base_file_name in existing_files:
        return None, f"File '{base_file_name}' already exists."

    # Save new file
    saved_file_path = os.path.join(f"user_files/{userid}/docs", base_file_name)
    shutil.move(file.name, saved_file_path)

    return saved_file_path, None


def save_files(files, userid):
    # Save files to user_files/userid/docs
    # Get only new added file paths
    # Write to user_files/userid/file_list.txt
    # Return list of only new added file paths
    added_file_paths = []
    already_exist_messages = []
    existing_files = get_user_existing_files_list(userid)

    for file in files:
        file_path, message = save_file(file, userid, existing_files)
        if file_path:
            added_file_paths.append(file_path)
        if message:
            already_exist_messages.append(message)

    # Update file list
    with open(f"user_files/{userid}/file_list.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join([os.path.basename(fp) for fp in added_file_paths + existing_files]))

    return added_file_paths, already_exist_messages


def process_file_and_load_user_profile(files, userid):
    global process_status
    global qa
    global USERID

    USERID = userid

    if userid == "":
        process_message = "请先输入用户ID -- Please enter a user ID first"
        return process_message

    else:
        existing_user = os.path.exists(f"user_files/{userid}")
        if existing_user:
            if files is not None:
                added_file_paths, _ = save_files(files, userid)
                vectordb = load_and_add_new_files_to_user_db(added_file_paths, userid)
            else: # no new files
                vectordb = load_user_db(userid)
            qa = create_qa_chain(vectordb)
            process_status = True
            process_message = "文件已分析完毕 Files have been processed."

        else: # new user
            if files is not None:
                # save files and create user db
                added_file_paths, _ = save_files(files, userid)
                process_message, vectordb = create_user_vectordb_with_initial_files(added_file_paths, userid)
                qa = create_qa_chain(vectordb)
                process_status = True
            else:
                process_message = "请先上传文件 Please upload a file first."
                return process_message

    return process_message


def get_answer(question, userid):
    global qa
    global process_status
    global USERID
    USERID = userid

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
    btn_process_and_load_user_profile = gr.Button("分析文件加载用户资料 Process files and load user profile")
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

    # btn_process_and_load_user_profile.click(fn=save_file_and_load_db, inputs = [pdf_upload], outputs = [output_question, output_answer])
    btn_process_and_load_user_profile.click(fn=process_file_and_load_user_profile, inputs = [pdf_upload, userid], outputs = [process_message])
    btn_ask.click(fn=get_answer, inputs = [current_question, userid], outputs = [current_answer, chat_hitsory, source_documents, generated_question])
    btn_clear.click(fn=clear_all_files_only, outputs = [pdf_upload, chat_hitsory, source_documents])
    btn_delete_user.click(fn=delete_user, inputs = [userid], outputs = [pdf_upload, chat_hitsory, source_documents, generated_question])

gr.close_all()
demo.launch(share=False, server_port=7878, allowed_paths=["D:\MyDocuments\03-PythonProjects\ChatGPT_Projects\chatpdf-demo\doc"])