from kivymd.color_definitions import colors
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.dialog import MDDialog


class ErrorDialog:
    def __init__(self, message: str):
        self.dialog = None
        self.title = 'Error'
        self.message = message
        self.show_dialog()

    def show_dialog(self):
        if not self.dialog:
            self.dialog = MDDialog(
                title=f'[color=%s]{self.title}[/color]' % colors['Red']["A700"],
                text=self.message,
                buttons=[
                    MDRaisedButton(
                        text="Ok",
                        on_release=lambda _: self.dialog.dismiss()
                    ),
                ]
            )
        self.dialog.open()
