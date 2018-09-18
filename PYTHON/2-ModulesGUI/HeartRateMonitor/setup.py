# settings for compilation of the entire program into a .app file

from setuptools import setup, find_packages

import sys

sys.setrecursionlimit(1500)

APP = ['bpm_monitor.py']
DATA_FILES = ['haarcascade_frontalface_alt.xml']
OPTIONS = {'argv_emulation': True}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
    name = 'Heart Rate Monitor',
    author = 'Abuzar Nur - Nihal Noor',
    author_email = 'anur10@student.monash.edu',
    description = 'ECE4095: Final Year Project - Visual Based Heart Rate Monitor'
   #  packages=find_packages(),
   # install_requires=['python-dateutil', 'pyserial']
)
