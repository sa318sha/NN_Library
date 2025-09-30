from PIL import Image, ImageOps
import os
import shutil
import glob
import random
import numpy as np


print(os.getcwd())
os.chdir('../Data/Images')
print(os.getcwd())


height =224
width = 224

img = Image.open('67.jpg')

img = img.resize((width,height),Image.ANTIALIAS)

img = ImageOps.grayscale(img)

arrayImage = np.array(img)

img.show()

print(img)
print(arrayImage)
# if os.path.isdir('train/dog') == False:
#     os.makedirs('train/dog')
#     os.makedirs('train/cat')
#     os.makedirs('valid/dog')
#     os.makedirs('valid/cat')
#     os.makedirs('test/dog')
#     os.makedirs('test/cat')
    
#     for c in random.sample(glob.glob('cat*'),500):
#         shutil.move(c, 'train/cat')
#     for c in random.sample(glob.glob('dog*'),500):
#         shutil.move(c, 'train/dog')
#     for c in random.sample(glob.glob('cat*'),100):
#         shutil.move(c, 'valid/cat')
#     for c in random.sample(glob.glob('dog*'),100):
#         shutil.move(c, 'valid/dog')
#     for c in random.sample(glob.glob('cat*'),50):
#         shutil.move(c, 'test/cat')
#     for c in random.sample(glob.glob('dog*'),50):
#         shutil.move(c, 'test/dog')
    

# print('pil version', PIL.__version__)
# main_data_path = 'NN_library/Kaggle_data'
# print(os.getcwd())
# os.chdir('../Data/Images')
# print(os.getcwd())
# print('path',os.path)
# os.chdir('../../../Kaggle images')
# if os.path.isdir('train/dog') == False:
#     os.makedirs('train/dog')
#     os.makedirs('train/cat')
#     os.makedirs('valid/dog')
#     os.makedirs('valid/cat')
#     os.makedirs('test/dog')
#     os.makedirs('test/cat')
#     for c in random.sample(glob.glob('cat*'),500):
#         shutil.move(c, 'train/cat')
#     for c in random.sample(glob.glob('dog*'),500):
#         shutil.move(c, 'train/dog')
#     for c in random.sample(glob.glob('cat*'),100):
#         shutil.move(c, 'valid/cat')
#     for c in random.sample(glob.glob('dog*'),100):
#         shutil.move(c, 'valid/dog')
#     for c in random.sample(glob.glob('cat*'),50):
#         shutil.move(c, 'test/cat')
#     for c in random.sample(glob.glob('dog*'),50):
#         shutil.move(c, 'test/dog')
    






# img = Image.open
# height, width = 224,224
# img = img.re
# print(glob)