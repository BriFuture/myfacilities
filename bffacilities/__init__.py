# -*- coding: utf-8 -*-

__version__ = "0.0.1alpha32"
__author__ = "BriFuture"

from ._constants import *

## For Translating

LANGUAGE_DIR = (Path(__file__).parent / "locale").resolve()
import gettext
def initGetText(domain="myfacilities") -> gettext.gettext:
    gettext.bindtextdomain(domain, LANGUAGE_DIR)
    gettext.textdomain(domain)
    gettext.find(domain, "locale", languages=["zh_CN", "en_US"])
    return gettext.gettext

from ._util import createLogger