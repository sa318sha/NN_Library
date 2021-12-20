import os
import numpy as np
from os.path import isfile,join
from PIL import Image, ImageOps
import pandas as pd


height =224
width = 224


class ImageDataGenerator():
    def __init__(self, preprocessing_function) -> None:
        pass

    def flow_from_directory(self,target_size, classes, shuffle=True, directory='/'):
        main_path = os.path.join(os.getcwd(),directory)

        main_array = []
        target_array = []
        for folder in os.listdir(main_path):
          for file in os.listdir(join(main_path,folder)):
            img = Image.open(join(main_path,folder,file)).resize((width,height),Image.ANTIALIAS)
            img = ImageOps.grayscale(img)
            main_array.append(np.array(img))
            target_array.append(folder)
          # print('length',len(folder))
          # array = np.array([np.array(Image.open(join(main_path,folder,file))) for file in os.listdir(join(main_path,folder))])/255
          # main_array = np.append(main_array,array)
          # #   print('hehe')
          # if folder == classes[1]:
          #   dogs = np.array([np.array(Image.open(join(main_path,folder,file))) for file in os.listdir(join(main_path,folder))])/255
        main_array = np.array(main_array)
        print(main_array.shape)
        main_array = main_array.reshape(main_array.shape[0],1,main_array.shape[1],main_array.shape[2])
        target_array = np.array(pd.get_dummies(target_array))
        # target_array = np.unique(target_array, return_inverse=True)
        print('target array,',target_array)
        print(main_array.shape,target_array.shape)




        if shuffle == True:
          shuffled_main = np.empty(main_array.shape, dtype=main_array.dtype)
          shuffled_target = np.empty(target_array.shape, dtype=target_array.dtype)
          permutation = np.random.permutation(len(main_array))
          for old_index, new_index in enumerate(permutation):
            shuffled_main[new_index] = main_array[old_index]
            shuffled_target[new_index] = target_array[old_index]
          print('shuffled main shape',shuffled_main.shape)
          # print('target array',shuffled_target)
          return shuffled_main/255, shuffled_target
        else:
          return main_array,target_array
        # print('array', array,array.shape)
        # print('main array',main_array.shape)
        # print('cats', cats)
        # print(type(cats))
        # print('dogs',dogs,type(dogs))
        # print('dogs',dogs.shape)
        # print('cats shape',cats.shape)
  

# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))] can use this for getting weights and biasis or this format more specifically
# train_batches() = ImageDataGenerator(None).flow_from_directory()



