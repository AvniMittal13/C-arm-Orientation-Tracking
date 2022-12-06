import math
import numpy as np

# function to get circumcenter -> return tvec of the circumcenter 
    # perpendicular bisectors of the 3 sides of a triangle meet at the point 
def getCircumcenter(A, B, C):
    """
        Input : 
            tvecs of center of 3 points -> center of aruco tag detectors 
            A,B,C -> translation vectors of the 3 points
             
        Output:
            tvec of circumcenter
    """
    # getting length of the sides
    a = np.linalg.norm(C - B)
    b = np.linalg.norm(C - A)
    c = np.linalg.norm(B - A)

    s = (a + b + c) / 2

    # circumradius 
    R = a*b*c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))

    # getting baycentric coordinates
    b1 = a*a * (b*b + c*c - a*a)
    b2 = b*b * (a*a + c*c - b*b)
    b3 = c*c * (a*a + b*b - c*c)

    # getting circumcenter
    P = np.column_stack((A, B, C)).dot(np.hstack((b1, b2, b3)))
    P /= b1 + b2 + b3

    return P 

def getRotAngleFromCenter(C, A, B):
    """
        Input:
            (C) center -> circumcenter
            (A) static -> pint on C-arm which is not moving
            (B) moving -> point on C-arm movies with C-arm
        Output:
            thetaC -> angle opposite to center C-arm

        Calculating using cosine formula
    """
    A = np.array(A[0][0])
    B = np.array(B[0][0])
    C = np.array(C[0])

    # print(A, B, C)

    ac = A - C
    bc = B - C

    # print(ac, bc)

    cosine_angle = np.dot(ac, bc) / (np.linalg.norm(ac) * np.linalg.norm(bc))
    thetaC = np.degrees(np.arccos(cosine_angle))
    print(cosine_angle)
    print(thetaC)
    return thetaC

    
