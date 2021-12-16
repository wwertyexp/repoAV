import os
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


if __name__ == '__main__':

    padded = np.zeros((800, 800, 3), dtype=np.uint8)


    images, labels = load_dataset()
    images, labels = shuffle(images, labels, random_state=42)
    images, labels = images[:1000],labels[:1000]

    folderBeard = "goodForBeard"

    if not os.path.exists(folderBeard):
        os.mkdir(folderBeard)

    beard_texture_base_img = '54_0_0_20170104211558436'

    textures = [
        cv2.imread(constants.BEARD_PNG_TEXTURES_FOLDER + '/' + f)
        for f in os.listdir(constants.BEARD_PNG_TEXTURES_FOLDER)
        if beard_texture_base_img in f
    ]

    new_textures = []
    for t in textures:
        p1 = t[0:5, 0:5, :]
        p2 = t[0:5, 5:10, :]
        p3 = t[5:10, 0:5, :]
        p4 = t[5:10, 5:10, :]
        new_textures.append(p1)
        new_textures.append(p2)
        new_textures.append(p3)
        new_textures.append(p4)
    textures = new_textures

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

        if beard_filter(f_name):

            # mod = add_attributes.add_beard(img, textures)
            padded[300:500, 300:500, :] = img
            z = add_attributes.add_beard(padded, textures)
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

    df = pd.DataFrame(data={'filename': filename, 'beard': beard, 'moustaches': moustache, 'glasses': glasses})
    df.to_csv(f"{name_csv}", index=False, header=False)
