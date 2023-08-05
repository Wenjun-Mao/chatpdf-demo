# gradio-app.py

import os
import shutil
import glob

from typing import IO

GRADIO_ANALYTICS_ENABLED = False
import gradio as gr
from utils import (
    prettify_source_documents,
    prettify_chat_history,
    create_user_vectordb_with_initial_files,
    load_user_db,
    load_and_add_new_files_to_user_db,
    create_qa_chain,
)

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
user_files_directory = os.environ["USER_FILES_DIRECTORY"]

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIAssistant:
    def __init__(self, userid):
        self.userid = userid
        self.qa = None
        self.process_status = False

    def get_user_existing_files_list(self):
        logger.info(f"Checking if user {self.userid} exists...")

        existing_files_basename = []

        # Check if user exists
        if os.path.exists(f"{user_files_directory}/{self.userid}"):
            try:
                # If user exists, read existing file names
                with open(
                    f"{user_files_directory}/{self.userid}/file_list.txt", "r", encoding="utf-8"
                ) as f:
                    existing_files_basename = f.read().splitlines()
            except FileNotFoundError:
                logger.info(
                    f"FileNotFoundError User {self.userid} file_list does not exist."
                )
                # Check if there are files in the "docs" folder
                docs_dir = f"{user_files_directory}/{self.userid}/docs"
                if os.path.exists(docs_dir) and glob.glob(docs_dir + "/*.pdf"):
                    logger.info("User list corrupted, reloading existing files...")
                    existing_files_basename = os.listdir(docs_dir)
                    logger.info(existing_files_basename)

                    # Delete existing db
                    if os.path.exists(f"{user_files_directory}/{self.userid}/chroma/"):
                        shutil.rmtree(f"{user_files_directory}/{self.userid}/chroma/")
                    existing_files_fullpath = [
                        os.path.join(docs_dir, f) for f in existing_files_basename
                    ]
                    _ = create_user_vectordb_with_initial_files(
                        existing_files_fullpath, self.userid
                    )

                    # Create file list
                    with open(
                        f"{user_files_directory}/{self.userid}/file_list.txt", "w", encoding="utf-8"
                    ) as f:
                        f.write("\n".join(existing_files_basename))
        logger.info(f"Existing files: {existing_files_basename}")
        return existing_files_basename

    def save_file(
        self, file: IO, existing_files_basename: list[str]
    ) -> tuple[str, str]:
        # if file already exists
        # return (None, "file already exists")
        # if file does not exist
        # save file and
        # return (file_path, None)
        if not os.path.exists(f"{user_files_directory}/{self.userid}/docs"):
            os.makedirs(f"{user_files_directory}/{self.userid}/docs")

        base_file_name = os.path.basename(file.name)

        # Check if file already exists in the list
        if base_file_name in existing_files_basename:
            return None, f"File '{base_file_name}' already exists."

        # Save new file
        saved_file_fullpath = os.path.join(
            f"{user_files_directory}/{self.userid}/docs", base_file_name
        )
        shutil.move(file.name, saved_file_fullpath)

        return saved_file_fullpath, None

    def save_files(self, files):
        # Save files to {user_files_directory}/userid/docs
        # Get only new added file paths
        # Write to {user_files_directory}/userid/file_list.txt
        # Return list of only new added file paths
        added_files_fullpaths = []
        already_exist_messages = []
        existing_files_basename = self.get_user_existing_files_list()

        for file in files:
            saved_file_fullpath, message = self.save_file(file, existing_files_basename)
            if saved_file_fullpath:
                added_files_fullpaths.append(saved_file_fullpath)
            if message:
                already_exist_messages.append(message)

        # Update file list
        with open(
            f"{user_files_directory}/{self.userid}/file_list.txt", "w", encoding="utf-8"
        ) as f:
            f.write(
                "\n".join(
                    [
                        os.path.basename(fp)
                        for fp in added_files_fullpaths + existing_files_basename
                    ]
                )
            )

        return added_files_fullpaths, already_exist_messages

    def process_file_and_load_user_profile(self, files):
        if self.userid == "":
            process_message = "请先输入用户ID -- Please enter a user ID first"
            return process_message

        else:
            existing_user = os.path.exists(f"{user_files_directory}/{self.userid}")
            if existing_user:
                if files is not None:
                    logger.info(f"Existing user, file not None")
                    added_file_fullpaths, _ = self.save_files(files)
                    logger.info(f"Added files: {added_file_fullpaths}")
                    logger.info(f"Current working directory: {os.getcwd()}")
                    if added_file_fullpaths == []:
                        vectordb = load_user_db(self.userid)
                    else:
                        vectordb = load_and_add_new_files_to_user_db(
                            added_file_fullpaths, self.userid
                        )
                else:  # no new files
                    logger.info(f"Existing user, file None")
                    vectordb = load_user_db(self.userid)
                logger.info(f"Crate qa chain with vectordb")
                self.qa = create_qa_chain(vectordb)
                self.process_status = True
                process_message = "文件已分析完毕 Files have been processed."

            else:  # new user
                if files is not None:
                    # save files and create user db
                    added_file_fullpaths, _ = self.save_files(files)
                    process_message, vectordb = create_user_vectordb_with_initial_files(
                        added_file_fullpaths, self.userid
                    )
                    self.qa = create_qa_chain(vectordb)
                    self.process_status = True
                else:
                    process_message = "请先上传文件 Please upload a file first."
                    return process_message

        return process_message

    def get_answer(self, question):
        if self.process_status:
            result = self.qa({"question": question})
            logger.info(result["answer"])
            logger.info(result)
            return (
                result["answer"],
                prettify_chat_history(result),
                prettify_source_documents(result),
                result["generated_question"],
            )

        else:
            error_msg = "请先上传并分析文件 Please upload and process a file first."
            return error_msg, error_msg, error_msg, error_msg


class GradioApp:
    def __init__(self):
        self.demo = gr.Blocks()
        self.ai_assistant = None

    def launch(self):
        with self.demo:
            gr.Markdown("# AI 论文小助手")
            userid = gr.Textbox(label="用户ID")
            pdf_upload = gr.Files(label="上传PDF文件")
            btn_process_and_load_user_profile = gr.Button(
                "分析文件加载用户资料 Process files and load user profile"
            )
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

            btn_process_and_load_user_profile.click(
                fn=self.process_file_and_load_user_profile,
                inputs=[pdf_upload, userid],
                outputs=[process_message],
            )
            btn_ask.click(
                fn=self.get_answer,
                inputs=[current_question],
                outputs=[
                    current_answer,
                    chat_hitsory,
                    source_documents,
                    generated_question,
                ],
            )
            btn_clear.click(
                fn=self.clear_conv_hsitory,
                outputs=[
                    current_question,
                    generated_question,
                    current_answer,
                    chat_hitsory,
                    source_documents,
                ],
            )
            btn_delete_user.click(
                fn=self.delete_user, inputs=[userid], outputs=[process_message]
            )

        gr.close_all()
        self.demo.launch(share=False, server_port=7878)

    def process_file_and_load_user_profile(self, files, userid):
        self.ai_assistant = AIAssistant(userid)
        process_message = self.ai_assistant.process_file_and_load_user_profile(files)
        return process_message

    def get_answer(self, question):
        (
            current_answer,
            chat_hitsory,
            source_documents,
            generated_question,
        ) = self.ai_assistant.get_answer(question)
        return current_answer, chat_hitsory, source_documents, generated_question

    def clear_conv_hsitory(self):
        try:
            vectordb = load_user_db(self.ai_assistant.userid)
            self.ai_assistant.qa = create_qa_chain(vectordb)
        except:
            pass
        current_question = ""
        generated_question = ""
        current_answer = ""
        chat_hitsory = ""
        source_documents = ""
        return (
            current_question,
            current_answer,
            generated_question,
            chat_hitsory,
            source_documents,
        )

    @staticmethod
    def delete_user(userid):
        try:
            if userid != "":
                if os.path.exists(f"{user_files_directory}/{userid}"):
                    shutil.rmtree(f"{user_files_directory}/{userid}")
                    process_message = f"用户 {userid} 已删除 -- user {userid} deleted "
                else:
                    process_message = f"用户 {userid} 不存在 -- user {userid} does not exist"
            else:
                process_message = "没有输入用户名 -- No user-id entered"
        except:
            process_message = (
                f"用户 {userid} 是当前活跃用户， 无法删除 -- "
                f"user {userid} is the current active user, cannot be deleted"
            )
        return process_message


if __name__ == "__main__":
    app = GradioApp()
    app.launch()
