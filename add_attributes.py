import cv2
from constants import *
import landmarks
import os
import numpy as np
from random import sample
import imutils
import math

glasses_png = [
    (cv2.imread(os.path.join(GLASSES_PNG_FOLDER,f) , cv2.IMREAD_UNCHANGED),f)
    for f in sorted(os.listdir(GLASSES_PNG_FOLDER))
    if not f.startswith('_')
]

moustaches_png = [
    (cv2.imread(os.path.join(MOUSTACHES_PNG_FOLDER,f) , cv2.IMREAD_UNCHANGED),f)
    for f in sorted(os.listdir(MOUSTACHES_PNG_FOLDER))
    if not f.startswith('_')
]

# for i, (img, f) in enumerate(moustaches_png):
#     if '1.png'==f:
#         h,w,_=img.shape
#         stack = np.zeros((h, w//8, 4))
#         tot = np.hstack([stack, img,stack])
#         h,w,_=tot.shape

#         stack = np.zeros((h//5, w, 4))
#         tot=np.vstack([stack,tot,stack])
#         cv2.imwrite(MOUSTACHES_PNG_FOLDER+'/'+f, tot)
#         moustaches_png[i] = (tot, f)



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

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def img_rotate(img, angle_x, angle_y, angle_z):
    x = angle_x
    y = angle_y
    z = angle_z
    ax = float(x * (math.pi / 180.0)) #0
    ay = float(y * (math.pi / 180.0)) 
    az = float(z * (math.pi / 180.0)) #0
    
    rx   = np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,1,0],
                 [0,0,0,1]],np.float32)  #0

    ry   = np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1]],np.float32)

    rz   = np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1]],np.float32)  #0
    rx[1,1] = math.cos(ax) #0
    rx[1,2] = -math.sin(ax) #0
    rx[2,1] = math.sin(ax) #0
    rx[2,2] = math.cos(ax) #0
    
    ry[0,0] = math.cos(ay)
    ry[0,2] = -math.sin(ay)
    ry[2,0] = math.sin(ay)
    ry[2,2] = math.cos(ay)
    
    rz[0,0] = math.cos(az) #0
    rz[0,1] = -math.sin(az) #0
    rz[1,0] = math.sin(az) #0
    rz[1,1] = math.cos(az) #0
    
    r =rx.dot(ry).dot(rz) # if we remove the lines we put    r=ry
    proj3dto2d = np.array([ [200,0,img.shape[1]/2,0],
                        [0,200,img.shape[0]/2,0],
                        [0,0,1,0] ],np.float32)
    trans= np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,1,230],   #400 to move the image in z axis 
                 [0,0,0,1]],np.float32)
    
    proj2dto3d = np.array([[1,0,-img.shape[1]/2],
                      [0,1,-img.shape[0]/2],
                      [0,0,0],
                      [0,0,1]],np.float32)
    final = proj3dto2d.dot(trans.dot(r.dot(proj2dto3d)))
    dst = cv2.warpPerspective(img, final,(img.shape[1],img.shape[0]),None,cv2.INTER_LINEAR
                              ,cv2.BORDER_CONSTANT,(255,255,255))
    return dst

def get_main_color_png(img):
    num_non_transparent = np.sum(img[:,:,3]!=0)
    R = np.sum(img[:,:,0] * img[:,:,3]) / num_non_transparent
    G = np.sum(img[:,:,1] * img[:,:,3]) / num_non_transparent
    B = np.sum(img[:,:,2] * img[:,:,3]) / num_non_transparent

    return R,G,B

def add_glasses(img, glasses,add_stecche=False, _landmarks=None):
    """img is a color image"""

    if (_landmarks is None):
        _landmarks = landmarks.get_lendmarks(img)

    if not _landmarks: return img
    x0,y0 = _landmarks.part(17).x,_landmarks.part(17).y
    x16,y16 = _landmarks.part(26).x,_landmarks.part(26).y

    dist = int(np.linalg.norm([x0-x16,y0-y16])*1.3)

    #get main glasses color
    main_glasses_color = get_main_color_png(glasses)


    w = dist
    h = int(glasses.shape[0]/glasses.shape[1]*w)
    glasses = cv2.resize(glasses, (w,h),interpolation = cv2.INTER_AREA)


    glasses = cv2.GaussianBlur(glasses, (3,3),0)
    # glasses = cv2.resize(glasses, img.shape[:2])

    shape = shape_to_np(_landmarks)
    left_eye = shape[36:41+1,:].mean(axis=0)
    right_eye = shape[42:47+1,:].mean(axis=0)
    eyes_distance = np.linalg.norm(left_eye - right_eye)

    right_eye_nose_distance = np.linalg.norm(right_eye-shape[27])
    left_eye_nose_distance = np.linalg.norm(left_eye-shape[27])
    face_width = (np.linalg.norm(shape[0]-shape[16]))

    y_angle = int((-right_eye_nose_distance + left_eye_nose_distance)/face_width*120)
    glasses = img_rotate(glasses, 0,y_angle,0)

    eye_angle = np.arctan2((left_eye[1]-right_eye[1]),(left_eye[0]-right_eye[0]))
    glasses = imutils.rotate(glasses, angle=-(180+eye_angle*180/3.14))


    tot = overlay_transparent(img.copy(), glasses,
    _landmarks.part(27).x-int(glasses.shape[1]/2),
    _landmarks.part(27).y-int(glasses.shape[0]/2))

    main_glasses_color = (np.array(main_glasses_color)*0.5).astype('int8')
    main_glasses_color = tuple( int(x) for x in main_glasses_color)
    if np.abs(y_angle)>10 and add_stecche:
        thickness = 2
        if (right_eye_nose_distance > left_eye_nose_distance):
            cv2.line(tot,
                (shape[16]),
                (np.mean([shape[26],shape[45]], axis=0)).astype('int'),
                main_glasses_color,thickness, lineType=cv2.LINE_AA)
        else:
            cv2.line(tot,
                (shape[0]),
                (np.mean([shape[17],shape[36]], axis=0)).astype('int'),
                main_glasses_color,thickness,lineType=cv2.LINE_AA)
    return tot

def add_beard(img, textures, mouth_near=0.5, beard_strength=0.5, beard_existence=0.5, becco=False, _landmarks=None):
    if (_landmarks is None):
        _landmarks = landmarks.get_lendmarks(img)
    
    if _landmarks is None: return img
    h,w,_ = img.shape
    shape = shape_to_np(_landmarks)

    poly = []

    for i in range(2,15):
        poly.append(shape[i])

    for i,j in [
                (64,11),
                (55,10),
                (56,9),
                (57,8),
                (58,7),
                (59,6),
                (60,4)
            ]:
        a=mouth_near
        poly.append((a*shape[i] + (1-a)*shape[j]).astype('int'))
    
        
    total_beard = np.zeros_like(img)

    grid_size = textures[0].shape[0]
    for i in range(0,w//grid_size):
        for j in range(0,h//grid_size):
            total_beard[
                    j*grid_size:(j+1)*grid_size,i*grid_size:(i+1)*grid_size,:
                ]=sample(textures,k=1)[0]


    total_beard = cv2.GaussianBlur(total_beard, [5,5],0)

    mask = np.zeros_like(img)

    for i in range(len(poly)-1):
        cv2.line(mask, poly[i], poly[i+1], (0,255,0))
    cv2.line(mask, poly[0],poly[-1],(0,255,0))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cv2.fillPoly(mask, np.array([poly],dtype=np.int32), 255)
   
    if becco:
        mask_becco = np.zeros_like(img)
        poly_becco = []
        for x in (48,54,10,9,8,7,6):
            poly_becco.append(shape[x])
        for i in range(len(poly_becco)-1):
            cv2.line(mask_becco, poly_becco[i], poly_becco[i+1], (0,255,0))
        cv2.line(mask_becco, poly_becco[0],poly_becco[-1],(0,255,0))
        mask_becco = cv2.cvtColor(mask_becco, cv2.COLOR_BGR2GRAY)

        cv2.fillPoly(mask_becco, np.array([poly_becco],dtype=np.int32), 255)
        _,mask_becco = cv2.threshold(mask_becco,127,1,cv2.THRESH_BINARY)
        mask = mask*mask_becco
        

    mask_upper = np.zeros_like(mask)
    
    mask_upper_poly = []
    mask_upper_poly.append([w-1,shape[14][1]])
    mask_upper_poly.append(shape[14])
    for x in reversed(range(2,14)):
        mask_upper_poly.append(shape[x])
    mask_upper_poly.append([0,shape[2][1]])
    mask_upper_poly.append([0,h-1])
    mask_upper_poly.append([w-1,h-1])

    # for x in mask_upper_poly:
    #     cv2.circle(mask_upper, x,10,255//2)
    cv2.fillPoly(mask_upper, np.array([mask_upper_poly],dtype=np.int32), 255)

    mask_upper = 255-mask_upper
    mask = cv2.GaussianBlur(mask, [15,15], 100,sigmaY=10)
    
    _,mask_upper=cv2.threshold(mask_upper, 127, 1, cv2.THRESH_BINARY)
    mask = mask*mask_upper

    mask = mask * beard_strength/255
    _mask = np.zeros_like(img,dtype=np.float32)
    _mask[:,:,0]=mask
    _mask[:,:,1]=mask
    _mask[:,:,2]=mask

    mask = _mask

    total_beard = np.array(total_beard, dtype=np.float32)/255
    img = np.array(img, dtype=np.float32)/255
    res=beard_existence*(mask)*total_beard+(1-mask)*img

    # cv2.imshow('mask', 1.0-mask)
    # cv2.imshow('img',img)
    # cv2.imshow('(1-mask)*img',(1-mask)*img)
    # cv2.imshow('(mask)*total_beard',(mask)*total_beard)
    # cv2.imshow('tot',(mask)*total_beard+(1-mask)*img)

    return res

def add_moustache(img, moustache, _landmarks=None, a=0.3):
    if (_landmarks is None):
        _landmarks = landmarks.get_lendmarks(img)
    
    if _landmarks is None: return img
    h,w,_ = img.shape
    shape = shape_to_np(_landmarks)

    x0 = int(a*shape[48][0] + (1-a)*shape[4][0])
    y0 = shape[48][1]
    x1 = int(a*shape[54][0] + (1-a)*shape[12][0])
    y1 = shape[54][1]

    dist = int(np.linalg.norm([x0-x1,y0-y1]))

    #get main glasses color
    main_glasses_color = get_main_color_png(moustache)

    w = dist
    h = int(moustache.shape[0]/moustache.shape[1]*w)
    moustache = cv2.resize(moustache, (w,h),interpolation = cv2.INTER_AREA)

    moustache = cv2.GaussianBlur(moustache, (3,3),0)

    left_mouth = shape[48]
    right_mouth = shape[54]
    left_eye = shape[36:41+1,:].mean(axis=0)
    right_eye = shape[42:47+1,:].mean(axis=0)
    right_eye_nose_distance = np.linalg.norm(right_eye-shape[27])
    left_eye_nose_distance = np.linalg.norm(left_eye-shape[27])
    face_width = (np.linalg.norm(shape[0]-shape[16]))

    y_angle = int((-right_eye_nose_distance + left_eye_nose_distance)/face_width*120)
    moustache = img_rotate(moustache, 0,y_angle,0)

    mouth_angle = np.arctan2((left_mouth[1]-right_mouth[1]),(left_mouth[0]-right_mouth[0]))
    moustache = imutils.rotate(moustache, angle=-(180+mouth_angle*180/3.14))

    moustache_face_center = np.mean([shape[33], shape[51]], axis=0)
    tot = overlay_transparent(img.copy(), moustache,
        int(moustache_face_center[0])-int(moustache.shape[1]/2),
        int(moustache_face_center[1])-int(moustache.shape[0]/2))
    return tot


def pixelate(input, pix_w=16, pix_h=16):
    height, width = input.shape[:2]
    # Desired "pixelated" size
    w, h = pix_w,pix_h

    # Resize input to "pixelated" size
    temp = cv2.resize(input, (w, h), interpolation=cv2.INTER_LINEAR)

    # Initialize output image
    output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    return output


if __name__ =='__main__':
    # img = cv2.imread('test.jpg')
    # add_glasses(img)

    clahe = cv2.createCLAHE(
            clipLimit=0.2,
            tileGridSize=(10, 10)
        )
    beard_texture_base_img = '54_0_0_20170104211558436'


    textures = [
        cv2.imread(BEARD_PNG_TEXTURES_FOLDER+'/'+f)
        for f in os.listdir(BEARD_PNG_TEXTURES_FOLDER)
        if beard_texture_base_img in f
    ]

    new_textures = []
    for t in textures:
        p1 = t[0:5,0:5,:]
        p2 = t[0:5,5:10,:]
        p3 = t[5:10,0:5,:]
        p4 = t[5:10,5:10,:]
        new_textures.append(p1)
        new_textures.append(p2)
        new_textures.append(p3)
        new_textures.append(p4)
    textures=new_textures
    while True:
        vid = cv2.VideoCapture(0)
        ret, frame = vid.read()
        # frame = cv2.resize(frame, (int(200/480*640),200),interpolation = cv2.INTER_AREA)
        frame = cv2.resize(frame, (200,200),interpolation = cv2.INTER_AREA)
        frame = pixelate(frame, 150,150)
        frame = cv2.GaussianBlur(frame, (5,5),0)
        vid.release()
        if  ret is None: continue
        # for glass,filename in glasses_png:
        #         mod = add_glasses(frame, glasses=glass, add_stecche=False)
        #         cv2.imshow(filename,mod)

        # equalization
        conv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        y_ch = conv[:,:,2].copy()
        conv[:,:,2] = clahe.apply(y_ch)
        equalized = cv2.cvtColor(conv, cv2.COLOR_HSV2BGR)

        _landmarks = landmarks.get_lendmarks(frame)

        mod = equalized
        mod = add_glasses(mod, pixelate(glasses_png[8][0], 200,2*100), _landmarks=_landmarks)
        mod = add_moustache(mod, moustaches_png[0][0], _landmarks=_landmarks)
        mod = add_beard(mod,
                textures,
                beard_strength=1,
                beard_existence=0.3,
                becco=False,
                _landmarks=_landmarks)
        cv2.imshow('edit',mod)

        k=cv2.waitKey(1000)

        if k==ord('q'):
            break
    cv2.destroyAllWindows()