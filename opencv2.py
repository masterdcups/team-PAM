import cv2
from os import listdir
import numpy as np
from datetime import datetime


path = 'C:\\Users\\ravel\\Google Drive\\M2\\Projet\\amicorpus\\IS1000a\\video\\IS1000a.C.avi'

array = [] #On sauvegarde le nombre de visages detecté

def difference(oldFrame,newFrame):

    print("Difference : ")
    diff = cv2.absdiff(oldFrame,newFrame)
    print('diff : ',diff)
#    contours = cv2.findContours(diff,mode=RETR_LIST)
#    cv2.drawContours(contours)
#    cv2.imshow(contours)
    input("Suivant")


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def detect():

    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml') #On import les face pour la detection
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
    camera = cv2.VideoCapture(path)

    camera.set(cv2.CAP_PROP_FPS, 1) #On reduit le nombre de frame a 1 pour faciliter les calculs
    i = 0


    while(True):
        ret,frame = camera.read()
        # print(type(ret),type(frame),ret)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        #calcul de la difference

        try:
            difference(oldFrame,gray)
            # err = mse(oldFrame,gray)
            # print("erreur : ",err)
            # if err > 200:
            #     cv2.imshow("oldFrame",oldFrame)
            #     cv2.imshow("gray",gray)
            #     input('attente...')
        except NameError:
            pass


        array.append(len(faces))

        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0),2)
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.03,5, 0, (40, 40))

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh),(0, 255, 0), 2)


            # cv2.imshow("camera", frame) #cv2.imwrite('./vikings.jpg', img)
            pathRes = './contours/' + str(i) + '.jpg'
            print(pathRes , len(faces))
            cv2.imwrite(pathRes, img)
            i+=1
        # if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
        #     break

        print('Nombres de visages trouvés en moyenne :' , np.mean(array))

        oldFrame = gray

    camera.release()
    cv2.destroyAllWindows()




detect()