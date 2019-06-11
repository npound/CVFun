
#OpenCV module
import cv2
#os module for reading training data directories and paths
import os
#numpy to convert python lists to numpy arrays as it is needed by OpenCV face recognizers
import numpy as np
import time
import uuid

subjects = [];


def createSubjects():
    f = open("subjects.txt","a");
    f.close();

def getSubjects():
    createSubjects();
    f = open("subjects.txt", "r")
    for x in f:
        subjects.append(x)
    f.close();

def getSubject(index):
    return subjects[index];

def resolveMultiFaces(tuplefaces, name):

    img = []
    index = 1
    cv2.destroyAllWindows();

    for t in tuplefaces:
        if t is not None:
            draw_text(t,str(index),0,0)
            cv2.imshow(str(index), t)
            cv2.waitKey(10);
        index += 1
    



    print("Enter the number of the photo you are")
    selection= input();
    cv2.destroyAllWindows();
    return int(selection)-1;

#function to detect face using OpenCV
def detect_face(img,face_aligner):

    aligned_faces = face_aligner.align_face(img)
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')
    
    if aligned_faces is None:
         return None, None;

    faces = []

    for a in aligned_faces:
 #convert the test image to gray image as opencv face detector expects gray images
        gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

        cv2.imshow("test",gray);
        #let's detect multiscale (some images may be closer to camera than others) images
        #result is a list of faces
        fcs = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

        if(len(fcs) !=0):
            faces.append((fcs[0], gray))


    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None

    #extract the face area

    
    #return only the face part of the image
    grey_faces = []

    for f in faces:
    #extract the face area
        (x, y, w, h) = f[0]
        grey_faces.append(cv2.resize(f[1][y:y+w, x:x+h], (400, 400)) )

    return grey_faces

def rememeber_who_i_know(face_aligner):
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
    dirs = os.listdir("people")

    for dir_name in dirs:
        label = int(dir_name)

        subject_dir_path = "people/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            faces.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            labels.append(label)

    return faces, labels

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)



def documentPerson(name):
    f = open("subjects.txt","a");
    f.write(name+"\r");
    f.close();
    subjects.append(name);
    return len(subjects)-1;


def rememberPerson(name, cam, face_aligner):

    faces = []
    labels = []
    
    index = documentPerson(name)
    os.mkdir(os.path.join(os.getcwd(),"people",str(index)))

    frames = [];
    now = time.time()
    future = now + 2

    while  time.time() < future:
        retval, frame = cam.read();
        frames.append(frame);

        if len(frames) > 9:
            break;

        cv2.imshow("You",frame);
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;
        pass;

    for f in frames:
        face_tuples = detect_face(f,face_aligner)
        path = os.path.join(os.getcwd() ,'people',str(index), str(index)+str(uuid.uuid4())+'.png');
        if face_tuples is not None:
            #####TODO need to cycle through faces with user to choose right face to remember
            if len(face_tuples) > 1:
                tface = face_tuples[resolveMultiFaces(face_tuples,name)];
            else:
                tface = face_tuples[0];

            res = cv2.cv2.imwrite(path, tface)
            faces.append(tface)
            labels.append(index)
    return faces, labels

def predict(test_img, face_recognizer,face_aligner):
   
   
    if len(subjects)  == 0:
        getSubjects();
        
   
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    tupple_faces = detect_face(img,face_aligner)



    if tupple_faces is  None:
        return -1, test_img;

    predictions = [];

    cv2.destroyAllWindows();

    for f in tupple_faces:
        if f is not None:
            cv2.imshow(str(uuid.uuid4()),f)
            cv2.waitKey(5);
            personIndex, competance = face_recognizer.predict(f);
            predictions.append((personIndex, competance ,f))

    cv2.destroyAllWindows();

    if len(predictions) == 0:
        return -1, test_img;


    return  predictions, img;


def setupRetrain(i, frame,face_aligner):

    faces = []
    labels = []
 
    path = os.path.join(os.getcwd() ,'people',str(i), str(i)+str(uuid.uuid4())+'.jpg');
    res = cv2.cv2.imwrite(path, frame)
    faces.append(frame)
    labels.append(i)
    
    return faces, labels





           # if dir_name != '0':
           #     image_path = subject_dir_path + "/" + image_name
          #      image = cv2.imread(image_path)
          #      face, rect = detect_face(image);
          #     path = os.path.join(os.getcwd() ,'people',dir_name,dir_name+str(uuid.uuid4())+'.png');
          #      if face is not None:
           #         res = cv2.cv2.imwrite(path, face)
          #          faces.append(face)
           #         labels.append(dir_name)
           #         image_path = subject_dir_path + "/" + image_name
           #         image = cv2.imread(image_path)