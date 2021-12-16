from tqdm import tqdm
import os
import cv2
import random
import numpy as np
import pandas as pd
import add_attributes
from constants import *

random.seed(SEED)

# loading dataset

def load_dataset(dataset_folder='../utkface', labels_csv='../labels_prof.csv'):
    X, y = [], []
    for line in tqdm(open(labels_csv)):
        filename, beard, moustache, glasses = line.split(',')
        beard, moustache, glasses = int(beard), int(moustache), int(glasses)
        img = cv2.imread(os.path.join(dataset_folder, filename), cv2.IMREAD_UNCHANGED)
        X.append((img))
        y.append((beard, moustache, glasses,filename))

    _X = np.array(X)
    _y = np.array(y)
    assert (len(_X) == len(_y))
    del X
    del y
    return _X, _y

X,Y = load_dataset()

X=X[:900]
Y=Y[:900]

from sklearn.utils import shuffle
X, Y = shuffle(X, Y, random_state=42)

carica_tommaso = False
if carica_tommaso:
    x=X[:400]
    y=Y[:400]
else:
    # x=X[400:800]
    # y=Y[400:800]
    x=X[800:900]
    y=Y[800:900]

print('loaded', len(x))
male_glasses=[]
female_glasses=[]
for i in range (len(add_attributes.glasses_png)):
    gender=int(add_attributes.glasses_png[i][1].split('_')[0])
    if gender==0:
        male_glasses.append(add_attributes.glasses_png[i][0])
    elif gender==1:
        female_glasses.append(add_attributes.glasses_png[i][0])
    else:
        male_glasses.append(add_attributes.glasses_png[i][0])
        female_glasses.append(add_attributes.glasses_png[i][0])

ext='.jpg.chip.jpg'

padded=np.zeros((800,800,3), dtype=np.uint8)

beard=[]
moustache=[]
glass=[]
filename=[]
for i in range(len(x)):
    _x = x[i]
    if int(y[i][GLASSES]) == 0:
        splittedFilename=y[i][FILENAME].split('_')
        age=int(splittedFilename[0])
        
        if age >= 5:
            gender=int(splittedFilename[1])
            if gender==0:
                glasses = random.sample(male_glasses,k=1)[0]
            else:
                glasses = random.sample(female_glasses,k=1)[0]
                
            padded[300:500,300:500,:]=_x 
            z=add_attributes.add_glasses(padded, glasses)
            z=z[300:500,300:500,:]

            cv2.imshow('added', z)
            k=cv2.waitKey(0)
            if k==ord('y'):
                beard.append(y[i,BEARD])
                moustache.append(y[i,MOUSTACHE])
                glass.append(1)
                file=y[i,FILENAME].split('.')[0]+'_glasses'+ext
                filename.append(file)
                cv2.imwrite(GENERATED_GLASSES_FOLDER+'/'+file,z)
                print('salva')
            if k==ord('n'):
                print('scartato')
            if k==ord('q'):
                break

df=pd.DataFrame(data={'filename':filename, 'beard':beard,'moustaches':moustache,'glasses':glass})
df.to_csv(f"{GENERATED_GLASSES_CSV}", index=False, header=False,mode='a')
cv2.destroyAllWindows()