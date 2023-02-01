import numpy as np
import cv2
import sys
import time
import keyboard
from utils import * 
import pandas as pd

# setting up ids of aruco tags
ID_center = 0   # won't really have an id for this, so need 4 ids in total
ID_static = 1
ID_moving = 2
ID_up = 3
ID_down = 4

ID_h_up = 5
ID_h_down = 6

actualAngle = 15
position = 5

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

def displayCoordinates(frame, markerCorner, tvec, rvec):
    
    x_translate, y_translate, z_translate = tvec[0][0][0], tvec[0][0][1], tvec[0][0][2]
    x_rot, y_rot, z_rot = rvec[0][0][0], rvec[0][0][1], rvec[0][0][2]
    
    corners = markerCorner.reshape((4, 2))
    (topLeft, topRight, bottomRight, bottomLeft) = corners
    
    topRight = (int(topRight[0]), int(topRight[1]))
    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
    topLeft = (int(topLeft[0]), int(topLeft[1]))
    
    cv2.putText(frame, "z: " + str(z_translate), (bottomLeft[0], bottomLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "y: " + str(y_translate), (bottomLeft[0], bottomLeft[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "x: " + str(x_translate), (bottomLeft[0], bottomLeft[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "Position",(bottomLeft[0], bottomLeft[1] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, "z: " + str(z_rot), (bottomRight[0], bottomRight[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "y: " + str(y_rot), (bottomRight[0], bottomRight[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "x: " + str(x_rot), (bottomRight[0], bottomRight[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "Orientation",(bottomRight[0], bottomRight[1] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, dataset):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()


    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters)
    aruco_display(corners, ids, rejected_img_points, frame)
    markerSize = 7 # enter ground truth of marker size in cm
    AxesScalingFactor = 0.002
    # print(ids)

    datasetCols = ['Original_Angle', 'position', 'angle', 'error']


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
            # print("Rotation vector: ", rvec)
            # print("Translation vector: ", tvec)
            x_translate, y_translate, z_translate = tvec[0][0][0], tvec[0][0][1], tvec[0][0][2]
            id_x[ids[i,0]] = x_translate
            id_y[ids[i,0]] = y_translate
            id_z[ids[i,0]] = z_translate
            
            t_vecs[ids[i,0]] = tvec

            cv2.aruco.drawDetectedMarkers(frame, corners) 
            # displayCoordinates(frame, corners[i], tvec, rvec)     # calling function to display positoin and orientation coordinates
            scaledTvec = tvec 
            scaledTvec[0] = scaledTvec*AxesScalingFactor
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, scaledTvec, 0.01) 
            # cv2.drawFrameAxes(frame, rvec, scaledTvec, 0.01) 
    
    # if ids is not None:
    #     # print(id_x)
    #     # print(id_x.keys().type)
    #     if 2 in id_x and 0 in id_x:
    #         # print(f"vertical distance: {abs(id_y[2] - id_y[0])}")
    #         # print(f"horizontal distance: {abs(id_x[2] - id_x[0])}")

    #         cv2.putText(frame, "vertical distance: "+ str(abs(id_y[2] - id_y[0])),(20,20), cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.5, (0, 255, 0), 2)
    #         cv2.putText(frame, "horizontal distance: "+ str(abs(id_x[2] - id_x[0])),(20,60), cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.5, (0, 255, 0), 2)


    #         cv2.putText(frame, "distance: "+ str(np.sqrt((id_y[2] - id_y[0])**2 + (id_x[2] - id_x[0])**2)),(20,100), cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.5, (0, 255, 0), 2)
    #         cv2.putText(frame, "distance2: "+ str(abs(np.linalg.norm(t_vecs[2]-t_vecs[0]))),(20,140), cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.5, (0, 255, 0), 2)


    # if ids is not None:

    #     if ID_center in id_x and ID_moving in id_x and ID_static in id_x :

    #         # getting rotation angle
    #         circumCenterCoord = getCircumcenter(t_vecs[ID_static], t_vecs[ID_center], t_vecs[ID_moving])
    #         thetaC = getRotAngleFromCenter(circumCenterCoord, t_vecs[ID_static], t_vecs[ID_moving])

    #         cv2.putText(frame, "Center pt: "+ str(circumCenterCoord),(20,160), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5, (0, 255, 0), 2)
    #         cv2.putText(frame, "Rot angle: "+ str(thetaC),(20,180), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5, (0, 255, 0), 2)

    if ids is not None:

        if ID_moving in id_x and ID_static in id_x and ID_up in id_x and ID_down in id_x:

            d = dict.fromkeys(datasetCols)
            

            # getting rotation angle
            c = getCircumcenter4(t_vecs[ID_up], t_vecs[ID_down], t_vecs[ID_moving])
            
            circumCenterCoord = getCircumcenter2(t_vecs[ID_up], t_vecs[ID_down], t_vecs[ID_moving])
            # thetaC = getRotAngleFromPt(circumCenterCoord, t_vecs[ID_static], t_vecs[ID_moving])
            thetaC = getRotAngleFromPt(c, t_vecs[ID_static], t_vecs[ID_moving])

            dir = getDir(id_y[ID_static], id_y[ID_moving])

            cv2.putText(frame, "Rot angle: "+ str(90 + dir * thetaC),(20,20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
            # cv2.putText(frame, "Rot angle u-d-static: "+ str(getRotAngleFromPt(t_vecs[ID_moving], t_vecs[ID_up], t_vecs[ID_down])),(20,180), cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5, (0, 255, 0), 2)
            print(f"Rot angle: { 90 + dir * thetaC } degrees", )

            d['angle'] = 90 + dir * thetaC
            dataset.loc[len(dataset)] = d


        if ID_h_up in id_x and ID_h_down in id_x :

            # print(f"Vertical Distance: { verticalHeight(id_y[ID_h_up], id_y[ID_h_down]) - 12+0.5 } cm", )
            cv2.putText(frame, "Vertical Distance: "+ str(verticalHeight(id_y[ID_h_up], id_y[ID_h_down]) - 12+0.5),(20,40), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)


        if ID_h_up in id_x and ID_static in id_x :
            
            # print(f"Horizontal distance: {HorizontalDist(id_x[ID_h_up], id_x[ID_static])} cm", )
            cv2.putText(frame, "Horizontal Distance: "+ str(HorizontalDist(id_x[ID_h_up], id_x[ID_static])),(20,
            60), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

        # print("///////////////////////////////")

    return frame


    

aruco_type = "DICT_5X5_100"

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])

arucoParams = cv2.aruco.DetectorParameters_create()


# intrinsic_camera = np.array(((933.15867, 0, 657.59),(0,933.1586, 400.36993),(0,0,1)))
# distortion = np.array((-0.43948,0.18514,0,0))


intrinsic_camera = np.array(((916.7279917   , 0.  ,       583.54549376),
                             (0.      ,   917.08446165, 359.63170424),
                             ( 0.     ,      0.  ,         1.        )))
distortion = np.array(( 0.04814238 , 0.52114135, -0.0222943,  -0.02534582, -0.43832353))


# intrinsic_camera = np.array(((963.784088   ,  0.  ,       644.9484349),
#                              (0.      ,   973.58533458, 358.04710496),
#                              ( 0.     ,      0.  ,         1.        )))
# distortion = np.array((0.18889935,  0.00504424, -0.02151044,  0.0071916,   0.32126311))

# creating csv file to store informaiton
datasetCols = ['Original_Angle', 'position', 'angle', 'error']
dataset = pd.DataFrame(columns = datasetCols)


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


while cap.isOpened():
    
    ret, img = cap.read()
    
    output = pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion, dataset)

    cv2.imshow('Estimated Pose', output)

    key = cv2.waitKey(1) & 0xFF
    if len(dataset)>199 or keyboard.is_pressed('q'):

        dataset["Original_Angle"] = actualAngle
        dataset["position"] = position
        dataset["error"] = dataset["Original_Angle"]-dataset["angle"]
        dataset.to_csv("errorAnalysisDatasets/angle_"+str(actualAngle)+"_"+str(position)+".csv", index = False)
        break

cap.release()
cv2.destroyAllWindows()

