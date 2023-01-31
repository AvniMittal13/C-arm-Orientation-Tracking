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
    print(f"Radius: {R}, coord Circumcenter: {P}")
    return P 


# Vector3f a,b,c // are the 3 pts of the tri

# Vector3f ac = c - a ;
# Vector3f ab = b - a ;
# Vector3f abXac = ab.cross( ac ) ;

# // this is the vector from a TO the circumsphere center
# Vector3f toCircumsphereCenter = (abXac.cross( ab )*ac.len2() + ac.cross( abXac )*ab.len2()) / (2.f*abXac.len2()) ;
# float circumsphereRadius = toCircumsphereCenter.len() ;

# // The 3 space coords of the circumsphere center then:
# Vector3f ccs = a  +  toCircumsphereCenter ; // now this is the actual 3space location

def getCircumcenter2(A, B, C):
    ac = C-A
    ab = B-A 

    abXac = np.cross(ab, ac)

    len_ac = np.linalg.norm(ac)
    len_ab = np.linalg.norm(ab)
    len_abXac = np.linalg.norm(abXac)

    toCircumcenter = np.cross(abXac, ab)*len_ac + np.cross(ac, abXac)*len_ab / (2 * len_abXac)
    R = np.linalg.norm(toCircumcenter)
    CC = A + toCircumcenter
    print(f"Radius: {R}, coord Circumcenter: {CC}")
    return CC

def getCircumcenter3(A,B,C):
    x1, y1, z1 = A[0][0]
    x2, y2, z2 = B[0][0]
    x3, y3, z3 = C[0][0]

    # print(x1, y1, z1)

    a = np.array([[(x1-x2),(y1-y2),(z1-z2)], [(x3-x2),(y3-y2),(z3-z2)], [(x1-x3),(y1-y3),(z1-z3)]])
    b = 1/2 *np.array([x1*x1-x2*x2+y1*y1-y2*y2+z1*z1-z2*z2, x3*x3-x2*x2+y3*y3-y2*y2+z3*z3-z2*z2, x1*x1-x3*x3+y1*y1-y3*y3+z1*z1-z3*z3])
    if np.linalg.det(a)==0:
        return np.array([-1,-1,-1])
    x = np.linalg.solve(a, b)
    x = np.asarray(x)
    x = np.expand_dims(x, axis = 0)
    x = np.expand_dims(x, axis = 0)
    print(np.asarray(x))
    return np.asarray(x)

def equation_plane(A, B, C):

    x1, y1, z1 = A[0][0]
    x2, y2, z2 = B[0][0]
    x3, y3, z3 = C[0][0]

    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    return [a,b,c,d]

def getCircumcenter4(A, B, C):
    x1, y1, z1 = A[0][0]
    x2, y2, z2 = B[0][0]
    x3, y3, z3 = C[0][0]

    plane =  equation_plane(A, B, C)
    p = plane[0]
    q = plane [1]
    r = plane[2] 

    a = np.array([[(x1-x2),(y1-y2),(z1-z2)], [(x3-x2),(y3-y2),(z3-z2)], [p,q,r]])
    b = 1/2 *np.array([x1*x1-x2*x2+y1*y1-y2*y2+z1*z1-z2*z2, x3*x3-x2*x2+y3*y3-y2*y2+z3*z3-z2*z2, (x1+x2)*p + (y1+y2)*q + (z1+z2)*r])

    cc = np.linalg.solve(a, b)
    cc = np.expand_dims(cc, axis = 0)
    cc = np.expand_dims(cc, axis = 0)

    return cc

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
    # print(f"Cosine angle : {cosine_angle}, theta: {thetaC}")
    return thetaC

def getRotAngleFromPt(C, A, B):
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
    C = np.array(C[0][0])

    # print(A, B, C)

    ac = A - C
    bc = B - C

    # print(ac, bc)

    cosine_angle = np.dot(ac, bc) / (np.linalg.norm(ac) * np.linalg.norm(bc))
    thetaC = np.degrees(np.arccos(cosine_angle))
    print(f"Cosine angle : {cosine_angle}, theta: {thetaC}")
    return thetaC

def verticalHeight(A, B):
    return abs(A-B)

def HorizontalDist(A, B):
    return abs(A-B)

def getDir(y_static, y_moving):
    if y_moving > y_static:
        return -1
    else:
        return 1