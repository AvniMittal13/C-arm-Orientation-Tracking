import torch
import cv2
import numpy as np
import keyboard
import yolov5
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

regressor = load_model('Regression_Models/w1.h5')
model = yolov5.load('400epochs_moredata.pt')
  
# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image


labels = {0: "BACK",
          1: "DOWN", 
          2: "LOCK",
          3: "SCREEN",
          4: "UP"}

# for each class -> get the bounding box with maximum prob

def resize_img(img):
    height, width = img.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = height / width

    # Define the desired size
    desired_width = 640
    desired_height = 640

    # Calculate the new dimensions based on the aspect ratio
    if aspect_ratio > 1:
        # The image is taller than it is wide
        new_width = desired_width
        new_height = int(desired_width * aspect_ratio)
        if new_height > desired_height:
            new_height = desired_height
            new_width = int(desired_height / aspect_ratio)
    else:
        # The image is wider than it is tall
        new_height = desired_height
        new_width = int(desired_height / aspect_ratio)
        if new_width > desired_width:
            new_width = desired_width
            new_height = int(desired_width * aspect_ratio)

    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height))

    # Create a black image with the desired size
    black_img = np.zeros((desired_height, desired_width, 3), np.uint8)

    # Calculate the position to place the resized image
    x_offset = int((desired_width - new_width) / 2)
    y_offset = int((desired_height - new_height) / 2)

    # Copy the resized image onto the black image
    black_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_img

    return black_img

def preprocess_frame(frame):
    frame = frame/255
    frame = resize_img(frame)
    return frame
    

def object_detect(frame, ss):

    # Pre-process the image
    preprocessed_img = preprocess_frame(frame) # Pre-process the image to match the expected input of your model
    input_tensor = torch.tensor(preprocessed_img, dtype=torch.float32).unsqueeze(0)

    # perform inference
    results = model(frame)

    # inference with larger input size
    results = model(frame, size=1280)

    # inference with test time augmentation
    results = model(frame, augment=True)

    # parse results
    predictions = results.pred[0]

    predictions = results.pred[0]
    bboxes = predictions[:, :4] # x1, y1, x2, y2 
    categories = predictions[:, 5]
    scores = predictions[:, 4]

    # print(bboxes)
    # print(categories)

    # dictionary : key (label/category) -> scores, bbox, label(int)
    if categories is not None:
        cat_list = categories.tolist()
        # print(cat_list)
        cat_dict = dict.fromkeys(list(map(int, cat_list)))
        # print(cat_dict)

    

    for score, bbox, cat in zip(scores, bboxes, categories):
        label = cat.item()
        # print(score)
        if cat_dict[label] != None:
            if score > cat_dict[label][0]:
                # replacing
                cat_dict[label] = (score, bbox, label) 

        else:
           cat_dict[label] = (score, bbox, label)  

    # Draw the bounding boxes on the image
    for label in cat_dict:
        x1, y1, x2, y2 = cat_dict[label][1]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, labels[cat_dict[label][2]],
                    (int(x1)+20,int(y1) + 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    d = dict()
    if cat_dict is not None:
        datasetCols = ["up1", "up2", "up3", "up4", "down1", "down2", "down3", "down4", "lock1", "lock2", "lock3", "lock4", "actualAngle"] 
        if 1 in cat_dict and 2 in cat_dict and 4 in cat_dict:
# up1,up2,up3,up4,down1,down2,down3,down4,lock1,lock2,lock3,lock4,actualAngle
            test = []
            x1, y1, x2, y2 = cat_dict[4][1]
            test.append(x1.item())
            test.append(y1.item())
            test.append(x2.item())
            test.append(y2.item())


            x1, y1, x2, y2 = cat_dict[1][1]
            test.append(x1.item())
            test.append(y1.item())
            test.append(x2.item())
            test.append(y2.item())

            x1, y1, x2, y2 = cat_dict[2][1]
            test.append(x1.item())
            test.append(y1.item())
            test.append(x2.item())
            test.append(y2.item())

            test = np.array(test).reshape((1, len(test)))
             
            print(test)
            X_test = ss.transform(test)
            
            prediction = regressor.predict(X_test)
            print(prediction)
            cv2.putText(frame, str(prediction[0][0]),
                    (20,20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
            # dataset.loc[len(dataset)] = d 

    # results.show()

    return frame


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

df = pd.read_csv("combinedDataset/dataset_all_pos1.csv")
X = df.drop("actualAngle", axis=1).values
ss = StandardScaler()
X_train = ss.fit_transform(X)


while cap.isOpened():
    
    ret, img = cap.read()

    output = object_detect(img, ss)
    
    cv2.imshow('Detected Objects', output)

    key = cv2.waitKey(1) & 0xFF
    if  keyboard.is_pressed('q'):
        break

cap.release()
cv2.destroyAllWindows()