import os
import random

import cv2
import numpy as np
from sklearn.utils import shuffle
import add_attributes
import constants
from tqdm import tqdm
import pandas as pd

BEARD, MOUSTACHE, GLASSES, FILENAME = 0, 1, 2, 3


def beard_filter(img_name):
    # [age]_[gender]_[race]_[date&time].jpg
    parts = img_name.split('_')
    age = int(parts[0])
    gender = int(parts[1])
    return (gender == 0) and (age > 20)


def load_dataset(dataset_folder='utkface', labels_csv='labels.csv'):
    X, y = [], []
    for line in tqdm(open(labels_csv)):
        filename, beard, moustache, glasses = line.split(',')
        beard, moustache, glasses = int(beard), int(moustache), int(glasses)
        img = cv2.imread(os.path.join(dataset_folder, filename), cv2.IMREAD_UNCHANGED)
        X.append((img))
        y.append((beard, moustache, glasses, filename))
    _X = np.array(X)
    _y = np.array(y)
    assert (len(_X) == len(_y))
    del X
    del y
    return _X, _y

def age_estimation(age, texture_dictionary):
    min = 200," "
    keys = list(texture_dictionary.keys())
    random.shuffle(keys)
    for k in keys:
        k.split("_")
        age_in_list = int (k.split("_")[0])
        diff = np.abs(age_in_list - age)
        if(diff < min[0]):
            min = diff,k
    return min[1]



if __name__ == '__main__':

    padded = np.zeros((800, 800, 3), dtype=np.uint8)


    images, labels = load_dataset()
    images, labels = shuffle(images, labels, random_state=42)
    images, labels = images[600:620],labels[600:620]


    folderBeard = "goodForBeard"

    if not os.path.exists(folderBeard):
        os.mkdir(folderBeard)

    beard_texture_base_img = '54_0_0_20170104211558436'

    d = dict()
    for elem in os.listdir(constants.BEARD_PNG_TEXTURES_FOLDER):
        t = cv2.imread(constants.BEARD_PNG_TEXTURES_FOLDER + "/" + elem)
        p1 = t[0:5, 0:5, :]
        p2 = t[0:5, 5:10, :]
        p3 = t[5:10, 0:5, :]
        p4 = t[5:10, 5:10, :]
        index = elem.rfind("_")
        key = elem[:index]
        if(key not in d):
            d[key] = []

        d[key].append(p1)
        d[key].append(p2)
        d[key].append(p3)
        d[key].append(p4)

    print("loaded")

    beard = []
    moustache = []
    glasses = []
    filechanged = []

    i = 0

    for img in images:

        filename=labels[i,3]
        ext = '.jpg.chip.jpg'
        f_name = filename.split(ext)[0]
        age = int (f_name.split("_")[0])


        if beard_filter(f_name):

            cv2.imshow("original",img)

            # mod = add_attributes.add_beard(img, textures)
            padded[300:500, 300:500, :] = img
            key = age_estimation(age,d)
            mouth_near = np.random.uniform(0.2,0.8)
            beard_strength = np.random.uniform(0.4,0.6)
            beard_existance = np.random.uniform(0.4,0.6)
            becco = np.random.choice((True,False))
            z = add_attributes.add_beard(padded, d[key],mouth_near,beard_strength,beard_existance,becco)
            z = z[300:500, 300:500, :]
            z = z * 255.0
            z = z.astype(np.uint8)
            print(z)
            cv2.imshow('Added beard', z)

            print("Ti piace la foto? y/n")
            k = cv2.waitKey(0)

            while True:
                if k == ord('n'):
                    break
                if k == ord('y'):
                    file_path = f'{folderBeard}' + '/' + f_name + '_add_beard' + ext
                    cv2.imwrite(file_path, z)
                    beard.append(1)
                    moustache.append(labels[i, 1])
                    glasses.append(labels[i, 2])
                    filechanged.append(file_path.split('/')[1])
                    break
            cv2.destroyAllWindows()

        else:
            pass

        i = i + 1

    name_csv="GENERATED_BEARD.csv"
    df = pd.DataFrame(data={'filename': filechanged, 'beard': beard, 'moustaches': moustache, 'glasses': glasses})
    df.to_csv(f"{name_csv}", index=False, header=False, mode="a")
