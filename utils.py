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
            theta -> angle opposite to center C-arm

        Calculating using cosine formula
    """
    a = np.linalg.norm(C - B)
    b = np.linalg.norm(C - A)
    c = np.linalg.norm(B - A)

    # applying cosine law
    cosC = (a**2 + b**2 - c**2)/(2*b*c)

    # getting cos inverse and converting to degrees
    thetaC = math.degrees(math.acos(cosC))

    return thetaC 

