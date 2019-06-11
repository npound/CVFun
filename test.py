
#OpenCV module
import cv2
#os module for reading training data directories and paths
import os
#numpy to convert python lists to numpy arrays as it is needed by OpenCV face recognizers
import numpy as np
import opencv
import time
import facealign;


def rememberPerson(cam, face_aligner):
        name = input("I don't know know you, Who are you?")
        faces, labels = opencv.rememberPerson(name, cam,face_aligner)
        face_recognizer.update(faces, np.array(labels))
        print("Ok I'll remember you")

def verifyIdentitys(predictions , frame, cam, face_aligner):

    for prediction in predictions:

#personIndex, competance ,f 
        personIndex = prediction[0]
        competance = prediction[1]
        img = prediction[2];
        cv2.destroyAllWindows();
        cv2.imshow(opencv.getSubject(personIndex),img);
        cv2.waitKey(2);


        if competance < 30:
            name =  opencv.getSubject(personIndex);
            s = "Hello {0} {1}".format(name[:-1], competance);
            cv2.destroyAllWindows();
            print(s)

        else:
            name = opencv.getSubject(personIndex);
            cv2.imshow(name[:-1],img);
            cv2.waitKey(2);
            s = "Is  {0} there? (Y/N) {1} " .format(name[:-1],competance);
            confirmIdentity = input(s);
            cv2.destroyAllWindows();
            if confirmIdentity == 'Y' or confirmIdentity == 'y':
                faces, labels = opencv.setupRetrain(personIndex, img,face_aligner);
                face_recognizer.update(faces, np.array(labels));
            else:
                rememberPerson(cam,face_aligner);

def initialize(face_aligner):
    faces, labels = opencv.rememeber_who_i_know(face_aligner);
    if len(faces) != 0:
        face_recognizer.train(faces, np.array(labels));





#face_recognizer = cv2.face.EigenFaceRecognizer_create();
face_recognizer = cv2.face.LBPHFaceRecognizer_create();
#face_recognizer = cv2.face.FisherFaceRecognizer_create();

face_aligner = facealign.FaceAligner();


if(os.path.isdir(os.path.join(os.getcwd(), "people")) == False):
    os.mkdir(os.path.join(os.getcwd(), "people"));

initialize(face_aligner);

cam = cv2.VideoCapture(0)
time.sleep(3);
while True:

    retval, frame = cam.read()
    cv2.cv2.waitKey(1);
    if retval != True:
        raise ValueError("Can't read frame")
    try:
       cv2.imshow("You",frame);
       predictions, img = opencv.predict(frame, face_recognizer,face_aligner)





       cv2.imshow("You",img);
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break;
       
       if predictions == -1:
            print("I can't see anyone.")
       else:
           verifyIdentitys(predictions , frame, cam,face_aligner)

    
    except cv2.cv2.error:
        rememberPerson(cam,face_aligner);

    time.sleep(1)

cv2.destroyAllWindows();

#
#faces, labels = opencv.prepare_training_data()
#
#face_recognizer.train(faces, np.array(labels))
#face_recognizer
#print("Predicting images...")


#for x in range(2):
 #   cam = cv2.VideoCapture(0)
  #  retval, frame = cam.read()
#
#    if retval != True:
        #raise ValueError("Can't read frame")
#
    #if x == 0:
        #test_img1 = frame
    #else:
        #test_img2 = frame
#
##perform a prediction
#predicted_img1 = 
#predicted_img2 = 
#print("Prediction complete")
#
##display both images
#cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
#cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.waitKey(1)
#cv2.destroyAllWindows()


