import cv2
import numpy as np
import os
from tqdm import tqdm

def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

def make_sunglasses(glasses, color = (0,0,0)):

    grey = cv2.cvtColor(glasses, cv2.COLOR_BGRA2GRAY)
    _,grey = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV)

    im_in=grey
    th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV)
    
    # Copy the thresholded image.
    im_floodfill = im_th.copy()
    
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    

    im_floodfill = np.array(im_floodfill, dtype=np.float32)/255
    glasses = glasses.copy()

    for i in range(h):
        for j in range(w):
            if im_floodfill[i,j]>0:
                if glasses[i,j,3] < 128:
                    glasses[i,j] = [0,0,0,0]
            else:
                glasses[i,j] = color
    return glasses

def _test():
    glasses = cv2.imread('1_0.png', cv2.IMREAD_UNCHANGED)
    h,w,_=glasses.shape
    glasses = cv2.resize(glasses, (w//5,h//5),interpolation = cv2.INTER_AREA)
    cap = cv2.VideoCapture(0)
    _,back=cap.read()
    cap.release()
    sun = make_sunglasses(glasses)
    # cv2.imwrite('sunglasses.png',sun)

    cv2.imshow('sun',sun)
    cv2.imshow('back',overlay_transparent(back, sun.copy(), 0,0))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # _test()
    alphas = (240, 180, 90, 40,20)
    files = [x for x in os.listdir('.') 
                    if '.png' in x and not x.startswith('_') and not 'sun' in x]
    import pprint
    pprint.pprint(files)
    for file in tqdm(files):
        for a in alphas:
            glasses = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            sun = make_sunglasses(glasses, color=(0,0,0,a))
            _file=file.replace('.png','')
            cv2.imwrite(f'{_file}_sun_{a}.png', sun) 