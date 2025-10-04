import os
import zipfile

with zipfile.ZipFile('mnist-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('mnist_dataset')

print(os.listdir('mnist_dataset'))