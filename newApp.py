from kivy.clock import Clock
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from PIL import Image as PILImage
from typing import List
import cv2
from model import Model
from addObjectDialog import AddObjectDialog
from camera import Camera
from errorDialog import ErrorDialog
from utils import *


class HomeScreen(MDScreen):
    pass


class NewApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.class_names: List[str] = []
        self.class_image_counters: dict = {}
        self.model_is_trained = False
        self.camera = None
        self.model: Model = None
        self.auto_prediction_is_enabled = False
        self.auto_prediction_event = None
        self.reset_training_data()

    @staticmethod
    def reset_training_data():
        clear_folder(f'{TRAINING_DATA_FOLDER}/')
        create_folder(f'{TRAINING_DATA_FOLDER}/temp')

    def reset_values(self):
        self.class_names = []
        self.class_image_counters = {}
        self.model_is_trained = False
        self.auto_prediction_is_enabled = False


    def reset(self):
        self.reset_values()
        self.reset_training_data()
        self.remove_add_object_buttons()


    def remove_add_object_buttons(self) -> None:
        object_buttons = self.root.ids.object_button_box
        add_object_buttons = [button for button in object_buttons.children if isinstance(button, MDRectangleFlatButton)]
        for button in add_object_buttons:
            object_buttons.remove_widget(button)

    def build(self):
        self.title = 'MDZ Ruhr OWL - Camera Object Classification'
        self.icon = 'images/mdz_icon.jpg'
        return HomeScreen()

        # Method is called after initialization of the UI


    def get_available_cameras(self,limit=3):
        """Check available camera indices."""
        index = 0
        arr = []
        while index < limit:
            cap = cv2.VideoCapture(index, cv2.CAP_MSMF)  # or cv2.CAP_ANY
            if cap.isOpened():
                arr.append(str(index))
                cap.release()
            index += 1
        return arr

    def on_camera_select(self, text):
        print(f'Selected camera: {text}')
        self.root.ids.camera.index = int(text) if text.isdigit() else 0
        print(f'New camera index: {self.root.ids.camera.index}')
        self.root.ids.camera.play = True

    def on_start(self):
        Clock.schedule_once(self.initialize_camera, 0)

    def initialize_camera(self, dt):
        self.camera = Camera(self.root.ids.camera)

    @staticmethod
    def error_dialog(message: str):
        return ErrorDialog(message)

    def train_model(self):
        if not self.class_names:
            self.error_dialog("No objects have been added yet.")
            return

        class_image_counters_values = self.class_image_counters.values()
        self.model.train(class_image_counters_values, num_epochs=7)
        self.model_is_trained = True

    def predict(self):
        if not self.model_is_trained:
            self.error_dialog("Model is not trained yet.")
            return

        image = self.camera.get_frame()

        if image is None:
            self.error_dialog("Failed to capture frame.")
            return

        prediction = self.model.predict(image)
        prediction_class = self.class_names[prediction]
        self.update_prediction_class(prediction_class)
        return prediction_class

    def update_prediction_class(self, prediction_class: str) -> None:
        self.root.ids.prediction_class.text = prediction_class

    def open_add_object_dialog(self) -> None:
        AddObjectDialog(self.add_class)

    def add_class(self, class_name: str) -> None:
        if not class_name:
            return

        self.class_names.append(class_name)
        self.class_image_counters[class_name] = 0
        self.create_image_folder(class_name)
        self.create_class_button(class_name)

        num_classes = len(self.class_names)
        self.model = Model(num_classes=num_classes)

    def create_image_folder(self, class_name: str) -> None:
        class_path = self.class_path(class_name)
        create_folder(class_path)

    def create_class_button(self, class_name: str) -> None:
        object_buttons = self.root.ids.object_button_box
        new_object_button = MDRectangleFlatButton(
            text=f'Add image for {class_name}',
            size_hint=(1, None),
            height=50,
            pos_hint={'center_x': 0.5, 'center_y': 0.5},
            on_release=lambda _: self.save_new_picture(class_name),
        )
        object_buttons.add_widget(new_object_button)

    def save_new_picture(self, class_name) -> None:
        image: PILImage = self.camera.get_frame()
        image_path = self.image_path(class_name)
        print(f'Saving image to {image_path}')
        image.save(image_path)
        self.class_image_counters[class_name] += 1

    def image_path(self, class_name: str) -> str:
        class_index = self.class_names.index(class_name)
        image_index = self.class_image_counters[class_name]
        return image_path(class_index, image_index)

    def class_path(self, class_name: str) -> str:
        class_index = self.class_names.index(class_name)
        return class_path(class_index)

    def auto_predict(self) -> None:
        if self.auto_prediction_is_enabled:
            self.turn_off_auto_prediction()
            return
        self.turn_on_auto_prediction()

    def turn_on_auto_prediction(self):
        if not self.model_is_trained:
            self.error_dialog("Model is not trained yet.")
            return

        auto_prediction_button = self.root.ids.auto_prediction_button
        self.disable_all_buttons()
        auto_prediction_button.md_bg_color = PRIMARY_GREEN
        auto_prediction_button.disabled = False
        self.auto_prediction_is_enabled = True
        self.auto_prediction_event = Clock.schedule_interval(self.run_auto_prediction, 0.1)

    def run_auto_prediction(self, dt) -> None:
        if not self.auto_prediction_is_enabled:
            Clock.unschedule(self.auto_prediction_event)
            return
        self.predict()

    def turn_off_auto_prediction(self):
        auto_prediction_button = self.root.ids.auto_prediction_button
        self.enable_all_buttons()
        auto_prediction_button.md_bg_color = PRIMARY_RED
        self.auto_prediction_is_enabled = False

    def disable_all_buttons(self) -> None:
        disable_buttons(self.root)

    def enable_all_buttons(self) -> None:
        enable_buttons(self.root)


if __name__ == '__main__':
    NewApp().run()
