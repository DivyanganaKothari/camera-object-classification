import os
import shutil

from kivymd.uix.button import MDRectangleFlatButton, MDRaisedButton

from settings import *


def path_exists(path: str) -> bool:
    return os.path.exists(path)


def create_folder(path: str) -> None:
    if not path_exists(path):
        os.makedirs(path)


def clear_folder(path: str) -> None:
    if path_exists(path):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)


def class_path(class_index: int) -> str:
    return f'{TRAINING_DATA_FOLDER}/{class_index}'


def image_path(class_index: int, image_index: int) -> str:
    return f'{class_path(class_index)}/frame_{image_index}.jpg'


def is_button(widget) -> bool:
    return isinstance(widget, MDRectangleFlatButton) or isinstance(widget, MDRaisedButton)


def disable_buttons(widget) -> None:
    if is_button(widget):
        widget.disabled = True
    for child in widget.children:
        disable_buttons(child)


def enable_buttons(widget) -> None:
    if is_button(widget):
        widget.disabled = False
    for child in widget.children:

        enable_buttons(child)

        enable_buttons(child)


