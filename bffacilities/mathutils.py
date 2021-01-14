import numpy as np
from math import pi, sqrt, atan2, asin
Rad2Deg = 180 / pi
Deg2Rad = pi / 180
# import copy
# 四元数转为旋转矩阵，将 R 系转化到 b 系的旋转矩阵
def quat2RotMat(q, mat):
    """
    q: np.ndarray(4)
    mat: np.ndarray(3)
    """
    # q: SE_q
    # # b => R
    # # row 1
    # mat[0] = 1 - 2 * q[2] * q[2] - 2 * q[3] * q[3]
    # mat[1] = 2 * (- q[0] * q[3] + q[1] * q[2])
    # mat[2] = 2 * (q[0] * q[2] + q[1] * q[3])
    # # row 2
    # mat[3] = 2 * (q[0] * q[3] + q[1] * q[2])
    # mat[4] = 1 - 2 * q[1] * q[1] - 2 * q[3] * q[3]
    # mat[5] = 2 * (- q[0] * q[1] + q[2] * q[3])
    # # row 3
    # mat[6] = 2 * (- q[0] * q[2] + q[1] * q[3])
    # mat[7] = 2 * (q[0] * q[1] + q[2] * q[3])
    # mat[8] = 1 - 2 * q[1] * q[1] - 2 * q[2] * q[2]

    # R => b
    # row 1
    mat[0] = 1 - 2 * q[2] * q[2] - 2 * q[3] * q[3]
    mat[1] = 2 * (q[0] * q[3] + q[1] * q[2])
    mat[2] = 2 * (- q[0] * q[2] + q[1] * q[3])
    # row 2
    mat[3] = 2 * (- q[0] * q[3] + q[1] * q[2])
    mat[4] = 1 - 2 * q[1] * q[1] - 2 * q[3] * q[3]
    mat[5] = 2 * (q[0] * q[1] + q[2] * q[3])
    # row 3
    mat[6] = 2 * (q[0] * q[2] + q[1] * q[3])
    mat[7] = 2 * (- q[0] * q[1] + q[2] * q[3])
    mat[8] = 1 - 2 * q[1] * q[1] - 2 * q[2] * q[2]

def eulerFromQuat(euler, q):
    pass

def eulerFromRotMat(euler, mat):
    euler[0] = atan2(mat[5], mat[8]) * Rad2Deg  # roll
    if mat[2] > 1.0:
        mat[2] = 1.0
    if mat[2] < -1.0:
        mat[2] = -1.0
    euler[1] = -asin(mat[2]) * Rad2Deg   # pitch
    euler[2] = atan2(mat[1], mat[0]) * Rad2Deg 

def rotMatFromEuler(euler):
    """
    默认使用 北东地 坐标系： 
        Heading - Z Axis
        Pitch   - Y Axis
        Roll    - X Axis
    旋转坐标系得到的旋转矩阵 （Base 坐标系到 Earch 坐标系的旋转矩阵 M）
    原向量 左乘 M.T 及则得到新坐标系下的坐标，
    在新坐标系下的向量 左乘 M 得到原坐标系下的坐标

    rot: np.ndarray (3, 3)
    
    将 Euler 角取反后相当于，旋转坐标系，最终得到的 3d 坐标是新坐标系下的坐标
    """
    y, p, r = euler * Deg2Rad
    cY = np.cos(y)
    sY = np.sin(y)
    cP = np.cos(p)
    sP = np.sin(p)
    cR = np.cos(r)
    sR = np.sin(r)
    rot = np.zeros((3, 3))
    rot[0, 0] = cY * cP
    rot[0, 1] = - sY * cR + cY * sP * sR
    rot[0, 2] = sY * sR + cY * cR * sP
    rot[1, 0] = cP * sY
    rot[1, 1] = cY * cR + sY * sP * sR
    rot[1, 2] = - cY * sR + cR * sY * sP
    rot[2, 0] = - sP
    rot[2, 1] = cP * sR
    rot[2, 2] = cP * cR
    # print(rot)
    return rot

def rotMatFromEuler2(euler):
    """
    默认使用 东北天 坐标系： 
        Heading - Z Axis
        Pitch   - Y Axis
        Roll    - X Axis
    旋转向量，坐标系不变，
    rot: np.ndarray (3, 3)
    
    将 Euler 角取反后相当于，旋转坐标系，最终得到的 3d 坐标是新坐标系下的坐标
    """
    y, p, r = euler * Deg2Rad
    cY = np.cos(y)
    sY = np.sin(y)
    cP = np.cos(p)
    sP = np.sin(p)
    cR = np.cos(r)
    sR = np.sin(r)
    rot = np.zeros((3, 3))
    rot[0, 0] = cY * cP
    rot[0, 1] = - sY * cR + cY * sP * sR
    rot[0, 2] = sY * sR + cY * cR * sP
    rot[1, 0] = cP * sY
    rot[1, 1] = cY * cR + sY * sP * sR
    rot[1, 2] = - cY * sR + cR * sY * sP
    rot[2, 0] = - sP
    rot[2, 1] = cP * sR
    rot[2, 2] = cP * cR
    # print(rot)
    return rot

def rotateVecWithEuler(vec, euler):
    """
    rotate a vector with a euler angle
    """
    rot = rotMatFromEuler(euler)
    return np.matmul(rot, vec)

if __name__ == "__main__":
    euler = np.asarray([90, 0, 45], dtype=np.float)
    res = rotateVecWithEuler((1, 0, 0), euler * -1)
    print("vec", res)