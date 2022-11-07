Hello ,


you can run this code by just runing final_run.py



######################################  Approch  ################################################

1 . Problem statement:
Develop a model that is able to process the feed/stream from a mounted/fixed camera(like CCTV) and able to detect in real time if the camera is pointed somewhere else. To implement this you may record a video from your mobile camera and then change is to some other angle after a while. Also the model should not confuse of moving people, if any, in the frame to be a valid detection

2 . performance metrics : binary_logloss,confusion matrix


3 . performed various data augumentations techniques

4 . performed original_image - currec_image on every image and trained a CNN model on top of that

5 . saved the model

6 . future scope : we can try different ways , instead of performing original_image - currec_image 

    we can give both original image  and current image to the model




