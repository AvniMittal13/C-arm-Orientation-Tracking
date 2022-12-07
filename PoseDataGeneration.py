import numpy as np
import cv2
import sys
import time 
import pandas as pd
import keyboard

# setting up ids of aruco tags
ID_center = 0   # won't really have an id for this, so need 4 ids in total
ID_static = 1
ID_moving = 2
ID_up = 3
ID_down = 4

############################################
actualAngle = 90    # KEEP CHANGING VALUE ACCORDING TO DATASET CURATION
############################################

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def aruco_display(corners, ids, rejected, image):
    
	if len(corners) > 0:
		
		ids = ids.flatten()
		
		for (markerCorner, markerID) in zip(corners, ids):
			
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners
			
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
			
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
			
			cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)
			# print("[Inference] ArUco marker ID: {}".format(markerID))
			
	return image

def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, dataset):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()


    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters)
    aruco_display(corners, ids, rejected_img_points, frame)
    markerSize = 7 # enter ground truth of marker size in cm
    AxesScalingFactor = 0.002
    # print(ids)

    # making dictionary 
    datasetCols = ["static_x", "static_y", "static_z", "moving_x", "moving_y", "moving_z", "up_x", "up_y", "up_z", "down_x", "down_y", "down_z", "angleActual"]


    if ids is not None:
        id_list = ids.reshape((1, len(ids))).tolist()[0]
        # print(ids)
        # print(id_list)
        id_x = dict.fromkeys(id_list)
        id_y = dict.fromkeys(id_list)
        id_z = dict.fromkeys(id_list)
        t_vecs = dict.fromkeys(id_list)
        
    if len(corners) > 0:
        for i in range(0, len(ids)):
           
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], markerSize, matrix_coefficients,     # getting coordinates for all ids
                                                                       distortion_coefficients)

            # rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], markerSize)
            
            scale = markerSize/AxesScalingFactor

            x_translate, y_translate, z_translate = tvec[0][0][0], tvec[0][0][1], tvec[0][0][2]
            id_x[ids[i,0]] = x_translate
            id_y[ids[i,0]] = y_translate
            id_z[ids[i,0]] = z_translate
            
            t_vecs[ids[i,0]] = tvec

            cv2.aruco.drawDetectedMarkers(frame, corners) 
            scaledTvec = tvec 
            scaledTvec[0] = scaledTvec*AxesScalingFactor
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, scaledTvec, 0.01) 
            # cv2.drawFrameAxes(frame, rvec, scaledTvec, 0.01) 
    

    if ids is not None:

        if ID_moving in id_x and ID_static in id_x and ID_up in id_x and ID_down in id_x:

            d = dict.fromkeys(datasetCols)
            d["static_x"] = t_vecs[ID_static][0][0][0]
            d["static_y"] = t_vecs[ID_static][0][0][1]
            d["static_z"] = t_vecs[ID_static][0][0][2]

            d["moving_x"] = t_vecs[ID_moving][0][0][0]
            d["moving_y"] = t_vecs[ID_moving][0][0][1]
            d["moving_z"] = t_vecs[ID_moving][0][0][2]

            d["up_x"] = t_vecs[ID_up][0][0][0]
            d["up_y"] = t_vecs[ID_up][0][0][1]
            d["up_z"] = t_vecs[ID_up][0][0][2]

            d["down_x"] = t_vecs[ID_down][0][0][0]
            d["down_y"] = t_vecs[ID_down][0][0][1]
            d["down_z"] = t_vecs[ID_down][0][0][2]

            d["actualAngle"] = actualAngle

            # print(d)
            # print(dataset)

            dataset.loc[len(dataset)] = d

    return frame


    

aruco_type = "DICT_5X5_100"

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])

arucoParams = cv2.aruco.DetectorParameters_create()


intrinsic_camera = np.array(((933.15867, 0, 657.59),(0,933.1586, 400.36993),(0,0,1)))
distortion = np.array((-0.43948,0.18514,0,0))


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



datasetCols = ["static_x", "static_y", "static_z", "moving_x", "moving_y", "moving_z", "up_x", "up_y", "up_z", "down_x", "down_y", "down_z", "angleActual"]
dataset = pd.DataFrame(columns = datasetCols)

while cap.isOpened():
    
    ret, img = cap.read()
    
    output = pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion, dataset)

    cv2.imshow('Estimated Pose', output)

    key = cv2.waitKey(1) & 0xFF
    if len(dataset)>400 or keyboard.is_pressed('q'):

        dataset["actualAngle"] = actualAngle
        dataset.to_csv("dataArucoTags/angle_"+str(actualAngle)+".csv", index = False)
        break

cap.release()
cv2.destroyAllWindows()

