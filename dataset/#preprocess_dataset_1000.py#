import os
import os.path
import random
from shutil import copy2
from binarizeImage import binarizeImage

def create_folders(path):
    sets = ['train', 'validation', 'test']
    for s in sets:
        temp_path = os.path.join(path, str(s))
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        temp_path = os.path.join(temp_path, 'chars')
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        for i in range(32,123):
            directory = os.path.join(temp_path, str(i).zfill(3))
            if not os.path.exists(directory):
                os.makedirs(directory)


def create_dataset(source_path, dest_path):
    create_folders(dest_path)
    sets = [('train', 1000), ('validation', 300), ('test', 200)]
    for (s, n) in sets:
        for root, dirs, files in os.walk(os.path.join(source_path, s, 'chars')):
            for d in dirs:
                directory = os.path.join(root, d)
                if os.listdir(directory):  # check if directory is not empty
                    class_files = []
                    for f in os.listdir(directory):
                        imfile=os.path.join(directory, f)
                        binarizeImage(imfile,dest_path)
                        
        
homedir='/home/tc9/Escritorio/Untitled Folder/'
create_dataset(homedir+'conv-net-keras/dataset/dataset_1000/', homedir+'conv-net-keras/dataset/dataset_1000_preproc/')
