#!/usr/bin/env python3
# coding:utf-8

"""windows 平台的系统栏图标
"""

import webbrowser
import os
import ctypes
from infi.systray import SysTrayIcon
from ._util import createLogger
logger = createLogger('bff_win_tray')

import locale
lang_code, code_page = locale.getdefaultlocale()

from sys import platform
if platform != "win32":
    raise TypeError('Only available on Windows ')


prototype = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.wintypes.HWND, 
    ctypes.wintypes.LPCWSTR, ctypes.wintypes.LPCWSTR, ctypes.wintypes.UINT)
paramflags = (1, "hwnd", 0), (1, "text", "Hi"), (1, "caption", "Hello from ctypes"), (1, "flags", 0)
MessageBox = prototype(("MessageBoxW", ctypes.windll.user32), paramflags)

class WinTray():
    def __init__(self):
        icon_path = os.path.join(os.path.dirname(__file__), "imgs", "logo.ico")
        self.systray = SysTrayIcon(icon_path, "bf-facilities", 
            self.make_menu(), on_quit = self.on_quit)
            # , left_click=self.on_show, right_click=self.on_right_click)

        reg_path = r'Software\Microsoft\Windows\CurrentVersion\Internet Settings'
        self.before_quit = None
        # self.INTERNET_SETTINGS = winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path, 0, winreg.KEY_ALL_ACCESS)

    def on_right_click(self):
        self.systray.update(menu=self.make_menu())
        self.systray._show_menu()

    def make_menu(self):
        return (
            ('Open', None, self.on_show), 
            ('dialogs', None, self.dialog_yes_no)
            )

    def on_show(self, widget=None, data=None):
        self.show_control_web()

    def show_control_web(self, widget=None, data=None):
        host_port = 8098
        webbrowser.open("http://127.0.0.1:%s/" % host_port)

    def on_quit(self, widget, data=None):
        # module_init.stop_all()
        if self.before_quit is not None:
            self.before_quit()

    def serve_forever(self):
        self.systray.start()

    def dialog_yes_no(self, msg="msg", title="Title", data=None, callback=None):
        res = MessageBox(text = "msg", caption = "title", flags = 1)
        # Yes:1 No:2
        if callback:
            callback(data, res)
        return res

sys_tray = None

def main(arg):
    global sys_tray
    sys_tray = WinTray()
    ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
    sys_tray.serve_forever()

if __name__ == '__main__':
    main()
