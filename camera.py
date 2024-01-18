from kivy.uix.camera import Camera as KivyCamera
from PIL import Image as PILImage, Image

from kivy.uix.button import Button
from settings import TRAINING_DATA_FOLDER


class Camera:
    def __init__(self, camera: KivyCamera) -> None:
        self.camera = camera

    def get_frame(self) -> PILImage:
        self.camera.export_to_png(f'{TRAINING_DATA_FOLDER}/temp/image.png')
        image = PILImage.open(f'{TRAINING_DATA_FOLDER}/temp/image.png')
        image.thumbnail((150, 150), Image.LANCZOS)
        return image.convert('RGB')