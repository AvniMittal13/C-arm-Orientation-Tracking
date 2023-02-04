import torch
import cv2
import numpy as np
import keyboard
import yolov5

# model = yolov5.load('yolov5s.pt')

# or load custom model
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
# set image
# img = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'


# perform inference
# results = model(img)

# # inference with larger input size
# results = model(img, size=1280)

# # inference with test time augmentation
# results = model(img, augment=True)

# # parse results
# predictions = results.pred[0]
# boxes = predictions[:, :4] # x1, y1, x2, y2
# scores = predictions[:, 4]
# categories = predictions[:, 5]

# # show detection bounding boxes on image
# results.show()

# Load the weights from the ".pt" file
# state_dict = torch.load("model_weights.pt")


# Load the model
# state_dict = torch.load("100epochs.pt")
# model.load_state_dict(state_dict)
# model.eval()

def preprocess_frame(frame):
    frame = frame/255
    return frame

# def get_bounding_boxes(outputs, image_height, image_width, threshold=0.5):
#     bboxes = []
#     confidences = []
#     for i in range(outputs[0].shape[0]):
#         xcenter, ycenter, w, h = outputs[0][i]
#         conf = outputs[1][i].item()
#         if conf >= threshold:
#             xmin = (xcenter - w/2) * image_width
#             ymin = (ycenter - h/2) * image_height
#             xmax = (xcenter + w/2) * image_width
#             ymax = (ycenter + h/2) * image_height
#             bboxes.append([xmin, ymin, xmax, ymax])
#             confidences.append(conf)
#     return bboxes, confidences

# def get_bounding_boxes(outputs, image_height, image_width, threshold=0.5):

    

def object_detect(frame):

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

    print(bboxes)
    print(categories)

    # Run inference on the input image
    # with torch.no_grad():
    #     outputs = model(input_tensor)

    # Process the output tensors to get the bounding boxes
    # print(predictions)
    # if predictions.shape != 0:
    #     bboxes = get_bounding_boxes(predictions, 640, 640) # Write logic to extract bounding boxes from `outputs`

    # Draw the bounding boxes on the image
    for bbox, cat in zip(bboxes, categories):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, labels[cat.item()],
                    (int(x1)+20,int(y1) + 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
    
    results.show()

    # return frame

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)


while cap.isOpened():
    
    ret, img = cap.read()
    
    output = object_detect(img)

    # cv2.imshow('Estimated Pose', output)

    key = cv2.waitKey(1) & 0xFF
    if keyboard.is_pressed('q'):
        break

cap.release()
cv2.destroyAllWindows()