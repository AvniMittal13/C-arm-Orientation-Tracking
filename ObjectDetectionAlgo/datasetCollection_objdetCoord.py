import torch
import cv2
import numpy as np
import keyboard
import yolov5
import pandas as pd

model = yolov5.load('400epochs.pt')
  
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


def preprocess_frame(frame):
    frame = frame/255
    return frame
    

def object_detect(frame, dataset):

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

    print(bboxes)
    print(categories)

    # dictionary : key (label/category) -> scores, bbox, label(int)
    if categories is not None:
        cat_list = categories.tolist()
        print(cat_list)
        cat_dict = dict.fromkeys(list(map(int, cat_list)))
        print(cat_dict)

    

    for score, bbox, cat in zip(scores, bboxes, categories):
        label = cat.item()
        print(score)
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


    if cat_dict is not None:
        datasetCols = ["up1", "up2", "up3", "up4", "down1", "down2", "down3", "down4", "lock1", "lock2", "lock3", "lock4", "screen1", "screen2", "screen3", "screen4", "acturalAngle"] 
        if 3 in cat_dict and 1 in cat_dict and 2 in cat_dict and 4 in cat_dict:

            d = dict.fromkeys(datasetCols)

            x1, y1, x2, y2 = cat_dict[1][1]
            d["down1"] = x1
            d["down2"] = y1
            d["down3"] = x2
            d["down4"] = y2

            x1, y1, x2, y2 = cat_dict[2][1]
            d["lock1"] = x1
            d["lock2"] = y1
            d["lock3"] = x2
            d["lock4"] = y2

            x1, y1, x2, y2 = cat_dict[3][1]
            d["screen1"] = x1
            d["screen2"] = y1
            d["screen3"] = x2
            d["screen4"] = y2

            x1, y1, x2, y2 = cat_dict[4][1]
            d["up1"] = x1
            d["up2"] = y1
            d["up3"] = x2
            d["up4"] = y2

            # print(d)
            # print(dataset)

            dataset.loc[len(dataset)] = d 

    # results.show()

    return frame

actualAngle = 90
datasetCols = ["up1", "up2", "up3", "up4", "down1", "down2", "down3", "down4", "lock1", "lock2", "lock3", "lock4", "screen1", "screen2", "screen3", "screen4", "acturalAngle"] 
dataset = pd.DataFrame(columns = datasetCols)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)


while cap.isOpened():
    
    ret, img = cap.read()
    
    output = object_detect(img, dataset)

    # cv2.imshow('Estimated Pose', output)

    # key = cv2.waitKey(1) & 0xFF
    if len(dataset)>2 or keyboard.is_pressed('q'):

        dataset["actualAngle"] = actualAngle
        dataset.to_csv("obj_det_bbox_dataset/angle_"+str(actualAngle)+".csv", index = False)
        break

cap.release()
cv2.destroyAllWindows()