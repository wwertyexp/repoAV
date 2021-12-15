import cv2
import os
import random
from constants import SEED,BEARD_PNG_TEXTURES_FOLDER

def load_beard_faces():
    beard_path = 'beard_faces/beard'
    return [
        (cv2.imread(beard_path+'/'+filename), filename)
        for filename in os.listdir(beard_path)
    ]

def main():
    beard_faces = load_beard_faces()
    random.seed(SEED)
    random.shuffle(beard_faces)

    # cv2.imshow('g',beard_faces[0][0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    grid_size = 10
    h,w,_=beard_faces[0][0].shape
    cv2.namedWindow('original',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('original', 400,400)
    for img, filename in beard_faces[:20]:
        x=0
        
        for i in range(w//grid_size//2,w//grid_size,1):
            for j in range(0,w//grid_size, 1):
                patch = img[i*grid_size:(i+1)*grid_size,
                            j*grid_size:(j+1)*grid_size:,
                            :].copy()
                img_with_grid = img.copy()
                grid_color = [255//5]*3
                img_with_grid[:,::grid_size,:] = grid_color
                img_with_grid[::grid_size,:,:] = grid_color

                img_with_grid[i*grid_size:(i+1)*grid_size,
                            j*grid_size:(j+1)*grid_size:,
                            :] = grid_color
                cv2.imshow('original',img_with_grid)
                cv2.imshow('patch',patch)
                k = -1
                while (k!=13):
                    k=cv2.waitKey(0)
                    if k==ord('q'):
                        exit(0)
                    if k==ord('s'):
                        ext = '.jpg.chip.jpg'
                        base = filename.split(ext)[0]
                        file_path = f'{BEARD_PNG_TEXTURES_FOLDER}/{base}_{x}{ext}'
                        cv2.imwrite(file_path, patch)
                        print('saved:',file_path)
                        x+=1


if __name__=='__main__':
    main()
    cv2.destroyAllWindows()
    