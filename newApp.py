from kivy.clock import Clock
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from PIL import Image as PILImage
from typing import List
import cv2
from kivy.metrics import dp
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton
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
        self.max_pictures_per_class = 35
        self.auto_prediction_is_enabled = False
        self.auto_prediction_event = None
        self.reset_training_data()
        self.available_cameras = []
        # Pre-compute menu items
        self.menu_items = [
            {"text": str(i), "viewclass": "OneLineListItem", "on_release": lambda x=i: self.on_camera_select(x)} for i
            in self.available_cameras
        ]
        self.dropdown_menu = None
        self.classes_with_popup_shown = set()


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
        self.classes_with_popup_shown.clear()
        self.update_prediction_class("Prediction Class")

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

    def get_available_cameras(self, limit=3):
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

    def open_camera_menu(self, instance):
        # Add separators between items
        menu_items = []
        for i, camera in enumerate(self.available_cameras):
            menu_items.append({"text": f"Camera {camera}", "viewclass": "OneLineListItem",
                               "on_release": lambda x=camera: self.on_camera_select(x),
                               "height": dp(45),  # Set the height of the item
                               "divider": None if i == len(self.available_cameras) - 1 else "Full"
                               # Disable divider for the last item
                               })

        self.dropdown_menu = MDDropdownMenu(
            caller=instance,
            items=menu_items,
            width_mult=2,  # Adjust this value for the desired width
            max_height=dp(100),  # Adjust this value for the desired height
        )
        self.dropdown_menu.open()

    def on_camera_select(self, selected_camera):

        self.root.ids.camera.index = int(selected_camera)
        self.root.ids.camera.play = True
        print(f"Selected camera from menu: {selected_camera}")
        # Manually dismiss the dropdown menu after selecting an item
        if self.dropdown_menu:
            self.dropdown_menu.dismiss()

    def on_start(self):
        Clock.schedule_once(self.initialize_camera, 0)

    def initialize_camera(self, dt):
        self.available_cameras = self.get_available_cameras()
        self.root.ids.camera.index = int(self.available_cameras[-1])
        self.root.ids.camera.play = True
        self.camera = Camera(self.root.ids.camera)

    @staticmethod
    def error_dialog(message: str):
        return ErrorDialog(message)

    def train_model(self):
        if not self.class_names:
            self.error_dialog("No objects have been added yet.")
            return

        class_image_counters_values = self.class_image_counters.values()
        self.model.train(class_image_counters_values, num_epochs=10)
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

        # Check if the maximum number of pictures has been reached
        if self.class_image_counters[class_name] >= self.max_pictures_per_class:
            self.show_max_pictures_popup(class_name)

    def show_max_pictures_popup(self, class_name):
        if class_name not in self.classes_with_popup_shown:
            dialog = MDDialog(
                title="Enough Pictures Clicked!",
                text=f"You have captured {self.max_pictures_per_class} pictures for {class_name}.",
                buttons=[
                    MDFlatButton(
                        text="OK",
                        on_release=lambda x: self.dismiss_max_pictures_popup(dialog),
                    )
                ],
            )
            dialog.open()
            self.classes_with_popup_shown.add(class_name)

    def dismiss_max_pictures_popup(self, dialog):
        dialog.dismiss()

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
