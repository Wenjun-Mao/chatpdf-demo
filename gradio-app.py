import gradio as gr
import os
import shutil

from util import (
    prettify_source_documents,
    prettify_chat_history,
    delete_user,
    clear_all_files_only,
    load_db
)


qa = None
process_status = False
USERID = ""


def load_user_files_into_db(file_list, userid):
    pass


def initialize_user(userid):
    # Check if user exists
    if os.path.exists(f"user_files/{userid}"):
        # If user exists, read existing file names
        try:
            with open(f"user_files/{userid}/file_list.txt", 'r', encoding='utf-8') as f:
                existing_files = f.read().splitlines()
        except FileNotFoundError:
            # Check if there are files in the "docs" folder
            docs_dir = f"user_files/{userid}/docs"
            if os.path.exists(docs_dir):
                print("User list corrupted, reloading existing files...")
                existing_files = os.listdir(docs_dir)
                # Add them to the list and write them to file_list.txt
                load_user_files_into_db(existing_files, userid)
                with open(f"user_files/{userid}/file_list.txt", 'w', encoding='utf-8') as f:
                    f.write('\n'.join(existing_files))
            else:
                existing_files = []
    else:
        existing_files = []

    return existing_files


def save_file(file, userid, existing_files):
    if not os.path.exists(f"user_files/{userid}/docs"):
        os.makedirs(f"user_files/{userid}/docs")

    base_file_name = os.path.basename(file.name)

    # Check if file already exists in the list
    if base_file_name in existing_files:
        return None, f"File '{base_file_name}' already exists."

    saved_file_path = os.path.join(f"user_files/{userid}/docs", base_file_name)
    shutil.move(file.name, saved_file_path)

    return saved_file_path, None


def save_files(files, userid):
    file_paths = []
    messages = []
    existing_files = initialize_user(userid)

    for file in files:
        file_path, message = save_file(file, userid, existing_files)
        if file_path:
            file_paths.append(file_path)
        if message:
            messages.append(message)

    # Update file list
    with open(f"user_files/{userid}/file_list.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join([os.path.basename(fp) for fp in file_paths + existing_files]))

    return file_paths, messages


def save_files_and_load_db(files, userid):
    global qa
    file_paths = save_files(files, userid)
    # qa = load_db(file_path)
    print(file_paths)
    qa = 1
    return qa


def process_files(files, userid):
    if (files is not None) and (userid != ""):
        try:
            qa = save_files_and_load_db(files, userid)
        except:
            return "Error processing file."
        process_status = True
        return "文件处理完成 Processing complete."
    else:
        return "没有文件或用户 ID File not uploaded."


def load_user_profile(userid):
    with open(f"user_files/{userid}/file_list.txt", 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    existing_files = '\n\n'.join(lines)
    num_files = len(lines)
    message = (
        f"用户资料加载完成 User profile loaded. User ID: {userid}"
        f"\n\n已上传文件 Uploaded files, 数量 {num_files}\n\n"
        f"{existing_files}"
    )
    return message


def process_file_and_load_user_profile(files, userid):
    global process_status
    global qa
    global USERID

    USERID = userid
    if userid == "":
        process_message = "请先输入用户ID -- Please enter a user ID first"

    else:
        if files is not None:
            process_message = process_files(files, userid)

        process_message = load_user_profile(userid)

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