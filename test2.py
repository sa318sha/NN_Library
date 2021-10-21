import numpy as np

np.random.seed(0)

image = np.random.rand(2,2,3,3)

print('image',image)
# print(np.zeros((1,image.shape[1])))

new_image = np.zeros((2,2,5,5))


# print('np.c_',np.c_[np.zeros((image.shape[1],1)), image, np.zeros((image.shape[1],1))])
# new_image = np.c_[np.zeros((image.shape[1],1)), image, np.zeros((image.shape[1],1))]
# print(new_image)

# print('row size', new_image.shape[1])

for img in range(image.shape[0]):
  for channel in range(image.shape[1]):
    new_image[img,channel] = np.r_[np.zeros((1,image.shape[3]+2)),
                  np.c_[np.zeros((image.shape[2],1)),
                  image[img,channel],
                  np.zeros((image.shape[2],1))],
                  np.zeros((1,image.shape[3]+2))]




# print(np.zeros((1,new_image.shape[1])))
# new_image = np.r_[np.zeros((1,image.shape[1]+2)),
#                   np.c_[np.zeros((image.shape[0],1)),
#                   image,
#                   np.zeros((image.shape[0],1))],
#                   np.zeros((1,image.shape[1]+2))]
print(new_image)
# newimage = np.column_stack((image,np.zeros((1,image.shape[0]))))