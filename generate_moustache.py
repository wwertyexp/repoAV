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

# def age_estimation(age, texture_dictionary):
#     min = 200," "
#     keys = list(texture_dictionary.keys())
#     random.shuffle(keys)
#     for k in keys:
#         k.split("_")
#         age_in_list = int (k.split("_")[0])
#         diff = np.abs(age_in_list - age)
#         if(diff < min[0]):
#             min = diff,k
#     return min[1]



if __name__ == '__main__':

    padded = np.zeros((800, 800, 3), dtype=np.uint8)


    images, labels = load_dataset('../utkface','../labels_prof.csv')
    images, labels = shuffle(images, labels, random_state=42)
    images, labels = images[500:520],labels[500:520]


    folderMoustache = constants.GENERATED_MOUSTACHE_FOLDER

    if not os.path.exists(folderMoustache):
        os.mkdir(folderMoustache)

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
        gender = int (f_name.split("_")[1])

        has_beard = int(labels[i,BEARD])
        has_moustache = int(labels[i,MOUSTACHE])

        if (beard_filter(f_name) and (has_moustache==0)):

            # mod = add_attributes.add_beard(img, textures)
            cv2.imshow('originale',img)
            padded[300:500, 300:500, :] = img
            _moustache = random.sample(add_attributes.moustaches_png, k=1)[0][0]
            z = add_attributes.add_moustache(padded, _moustache)
            z = z[300:500, 300:500, :]
            # z = z * 255.0
            z = z.astype(np.uint8)
            print(z)
            cv2.imshow('Added moustache', z)

            print("Ti piace la foto? y/n")
            k = cv2.waitKey(0)

            while True:
                if k == ord('n'):
                    break
                if k == ord('y'):
                    beard.append(labels[i, BEARD])
                    glasses.append(labels[i, GLASSES])

                    file_path = f'{folderMoustache}' + '/' + f_name + '_add_moustache' + ext
                    if not has_beard:
                        moustache.append(1)
                    else:
                        moustache.append(0)

                    cv2.imwrite(file_path, z)
                    filechanged.append(file_path.split('/')[1])
                    break
            cv2.destroyAllWindows()

        else:
            pass

        i = i + 1

    name_csv=constants.GENERATED_MOUSTACHE_CSV
    df = pd.DataFrame(data={'filename': filechanged, 'beard': beard, 'moustaches': moustache, 'glasses': glasses})
    df.to_csv(f"{name_csv}", index=False, header=False, mode="a")
