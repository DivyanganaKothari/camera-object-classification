from kivy.uix.camera import Camera as KivyCamera
from PIL import Image as PILImage, Image

from settings import TRAINING_DATA_FOLDER


class Camera:
    def __init__(self, camera: KivyCamera) -> None:
        self.camera = camera

    def get_frame(self) -> PILImage:
        texture = self.camera.texture
        # Retrieve the image data from the texture
        data = texture.pixels

        # Create a PIL Image from the texture data
        image = PILImage.frombytes(mode='RGBA', size=texture.size, data=data)

        return image.convert('RGB')


 #       self.camera.export_to_png(f'{TRAINING_DATA_FOLDER}/temp/image.png')
 #       image = PILImage.open(f'{TRAINING_DATA_FOLDER}/temp/image.png')
 #       image.thumbnail((150, 150), Image.LANCZOS)
 #       return image.convert('RGB')
