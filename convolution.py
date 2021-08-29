import numpy as np
from numpy.random import randint
from PIL import Image,ImageOps

# set image path and dimension
im_path='image path'
img_size=300

# set kernel size
ker_size=3
min_val=-(ker_size-1)//2
max_val=(ker_size+1)//2

# load image and convert to grayscale (2d array)
im=Image.open(im_path)
im=im.resize((img_size,img_size))
im=ImageOps.grayscale(im)

# convert image array to list form
im_arr=np.array(im)
arr=[list(i) for i in im_arr]

# create kernel of randomly generated integers
ker=[]
while len(ker)<ker_size:
    ker_val=list(randint(min_val,max_val,ker_size))
    if sum(ker_val)==0:
        ker.append(ker_val)


def conv2d(arr, ker):
    """
    :param arr: 2d image array in list form
    :param ker: 2d kernel in list form
    :return: 2d array
    """

    conv = []  # for storing convolution output

    # iterate over the array
    for i in range(len(arr) - len(ker) + 1):
        for j in range(len(arr) - len(ker) + 1):
            temp = [] # for storing sub arrays

            # store sub array
            for k in range(len(ker)):
                temp.append(arr[i + k][j:j + len(ker)])

            # get value of the corresponding pixel
            res = 0
            for m in range(len(ker)):
                for n in range(len(ker)):
                    res += temp[m][n] * ker[m][n]

            # store the value
            conv.append(res)

    # find largest magnitude
    m = 0
    for i in conv:
        if m < abs(i):
            m = i

    # normalise the values and store them
    conv = [int((abs(i) / m) * 255) for i in conv]

    # set size for reshaping into 2d array
    size = int(len(conv) ** 0.5)

    return np.array(conv, np.uint8).reshape((size, size))


# display
Image.fromarray(conv2d(arr,ker),"L").show()



