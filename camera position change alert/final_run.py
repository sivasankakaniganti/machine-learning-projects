import joblib
import time
import cv2
import numpy as np
import warnings
import tensorflow as tf
from tensorflow.keras.models import load_model
warnings.filterwarnings("ignore")
m = load_model('model/cnn2.h5')
# m =joblib.load('model.pkl')
cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop) 
ret,main_image = cap.read()
cap.release()

vid = cv2.VideoCapture(0)
i = 0
while(True):
      
    # Capture the video frame
    # by frame
    ret, f = vid.read()
    # Display the resulting frame
    frame = main_image - f
    #frame = frame.flatten()
    val = m.predict(frame.reshape(1,480,640,3))
    #print(val)
    if val[0]>0.3:
        #red 
        val = [0,0,255]
    else:
        #green
        val = [0,255,0]
    f = cv2.copyMakeBorder(f,20,20,20,20,cv2.BORDER_CONSTANT,value=val)
    cv2.imshow('frame', f)
    #time.sleep(2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
