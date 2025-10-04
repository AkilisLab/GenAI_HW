import os
import shutil

source = os.path.expanduser('~/Downloads/kaggle.json')
destination = os.path.expanduser('~/.kaggle/kaggle.json')

os.makedirs(os.path.dirname(destination), exist_ok=True)
shutil.copyfile(source, destination)