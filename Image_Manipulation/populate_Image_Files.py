import os
import shutil
import glob
import random

from PIL import Image, ImageOps

import numpy as np

# always from inside the image_manipulation folder

print(os.getcwd())
os.chdir('../Data/processed_images')
image_directory = os.getcwd()
print(os.getcwd())
print('image directory',image_directory)
cat_train = '/train/cat'
cat_valid = '/valid/cat'
cat_test = '/test/cat'
dog_train = '/train/dog'
dog_valid = '/valid/dog'
dog_test = '/test/dog'



height =224
width = 224



if os.path.isdir('train/dog') == False:
  print('working')
  os.makedirs('train/dog')
  os.makedirs('train/cat')
  os.makedirs('valid/dog')
  os.makedirs('valid/cat')
  os.makedirs('test/dog')
  os.makedirs('test/cat')



  os.chdir('../../../Kaggle_images/Cat')
  print(os.getcwd())
  for c in random.sample(glob.glob('*.jpg'),500):
    img = Image.open(c).resize((width,height),Image.ANTIALIAS)
    img = ImageOps.grayscale(img)
    img.save(c)
    shutil.move(c, image_directory+cat_train)
  for c in random.sample(glob.glob('*.jpg'),100):
    img = Image.open(c).resize((width,height),Image.ANTIALIAS)
    img = ImageOps.grayscale(img)
    img.save(c)
    shutil.move(c, image_directory+cat_valid)
  for c in random.sample(glob.glob('*.jpg'),50):
    img = Image.open(c).resize((width,height),Image.ANTIALIAS)
    img = ImageOps.grayscale(img)
    img.save(c)
    shutil.move(c, image_directory+cat_test)

  os.chdir('../Dog')

  for c in random.sample(glob.glob('*.jpg'),500):
    img = Image.open(c).resize((width,height),Image.ANTIALIAS)
    img = ImageOps.grayscale(img)
    img.save(c)
    shutil.move(c, image_directory+dog_train)
  for c in random.sample(glob.glob('*.jpg'),100):
    img = Image.open(c).resize((width,height),Image.ANTIALIAS)
    img = ImageOps.grayscale(img)
    img.save(c)
    shutil.move(c, image_directory+dog_valid)  
  for c in random.sample(glob.glob('*.jpg'),50):
    img = Image.open(c).resize((width,height),Image.ANTIALIAS)
    img = ImageOps.grayscale(img)
    img.save(c)
    shutil.move(c, image_directory+dog_test)
    
  os.chdir('../../NN_library')
else:
  print('not working')
  pass

print(os.getcwd())
