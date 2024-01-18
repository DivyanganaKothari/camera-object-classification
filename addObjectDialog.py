from kivymd.uix.button import MDRaisedButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.textfield import MDTextField


class AddObjectDialog:
    def __init__(self, add_object_method):
        self.add_object_method = add_object_method
        self.object_name: str = None
        self.dialog: MDDialog = None
        self.text_field: MDTextField = None
        self.title: str = 'Add Object'
        self.ask_for_new_object_name()

    def ask_for_new_object_name(self) -> None:
        if not self.dialog:
            self.text_field = MDTextField(
                hint_text='Enter the name of the object',
                size_hint=(0.5, 0.5),
                multiline=False,
            )
            self.dialog = MDDialog(
                title=self.title,
                type='custom',
                size_hint=(0.5, 0.5),
                content_cls=self.text_field,
                buttons=[
                    MDRaisedButton(
                        text='Ok',
                        on_release=self.confirm_object_name,
                    ),
                    MDRaisedButton(
                        text='Cancel',
                        on_release=lambda _: self.dismiss()
                    ),
                ]
            )
        self.dialog.open()

    def confirm_object_name(self, obj):
        self.object_name = self.text_field.text
        self.dismiss()

    def dismiss(self):
        self.dialog.dismiss()
        self.add_object_method(self.object_name)
