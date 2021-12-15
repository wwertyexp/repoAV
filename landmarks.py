import dlib
import cv2
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor( 'shape_predictor_68_face_landmarks.dat' )

def get_lendmarks(frame, imgshow=False, figure='facial landmarks', show_numbers=True):
    '''Return the lendmarks of the given face (gray level image).
    Optionally, you can show the image with the lendmarks.
    Return None if no faces are detected in the image.
    '''

    grayframe = frame#cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(grayframe, 1)
    showface = frame.copy()
    if(len(faces)==0):
        #print('len(faces) =',len(faces))
        #cv2.imshow('error',showface)
        #cv2.waitKey()
        return None
        #raise Exception('no faces detected')
    for rect in faces:
        bbox = [ rect.left(), rect.top(), rect.right(), rect.bottom()]
        shape = predictor(grayframe, rect)

        if (imgshow):
            # draw points and point numbers
            c =  (0, 255, 0)
            for i in range(0,68):
                p = (shape.part(i).x, shape.part(i).y)
                if (show_numbers):
                    cv2.putText(img=showface, text=str(i), org=p,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX , fontScale=0.3, color=[0, 0, 0], lineType=cv2.LINE_AA, thickness=3)
                    cv2.putText(img=showface, text=str(i), org=p,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX , fontScale=0.3, color=[255, 255, 255], lineType=cv2.LINE_AA, thickness=1)
                cv2.circle(showface, p, 2, c, -1)
            print('imgshow')
            cv2.imshow(figure, showface)
        return shape


if __name__ =='__main__':
    img = cv2.imread('test.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    get_lendmarks(img,imgshow=True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()