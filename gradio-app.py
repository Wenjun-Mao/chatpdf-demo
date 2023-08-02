import gradio as gr
import os
import shutil

from util import prettify_source_documents, prettify_chat_history, load_db


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