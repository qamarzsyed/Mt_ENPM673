import numpy as np

# function to make each 3x12 portion of the A matrix
def A_stack(x, X):
    u, v, w = x[0], x[1], x[2]

    a_list = [[0, 0, 0, 0, -w * X[0], -w * X[1], -w * X[2], -w * X[3], v * X[0], v * X[1], v * X[2], v * X[3]],
              [w * X[0], w * X[1], w * X[2], w * X[3], 0, 0, 0, 0, -u * X[0], -u * X[1], -u * X[2], -u * X[3]],
              [-v * X[0], -v * X[1], -v * X[2], -v * X[3], u * X[0], u * X[1], u * X[2], u * X[3], 0, 0, 0, 0]
              ]
    return np.array(a_list)

# initiating all the image coords, world coords, and A stacks from those 2
x_1 = np.array([757, 213, 1])
x_2 = np.array([758, 415, 1])
x_3 = np.array([758, 686, 1])
x_4 = np.array([759, 966, 1])
x_5 = np.array([1190, 172, 1])
x_6 = np.array([329, 1041, 1])
x_7 = np.array([1204, 850, 1])
x_8 = np.array([340, 159, 1])

X_1 = np.array([0, 0, 0, 1])
X_2 = np.array([0, 3, 0, 1])
X_3 = np.array([0, 7, 0, 1])
X_4 = np.array([0, 11, 0, 1])
X_5 = np.array([7, 1, 0, 1])
X_6 = np.array([0, 11, 7, 1])
X_7 = np.array([7, 9, 0, 1])
X_8 = np.array([0, 1, 7, 1])

A_1 = A_stack(x_1, X_1)
A_2 = A_stack(x_2, X_2)
A_3 = A_stack(x_3, X_3)
A_4 = A_stack(x_4, X_4)
A_5 = A_stack(x_5, X_5)
A_6 = A_stack(x_6, X_6)
A_7 = A_stack(x_7, X_7)
A_8 = A_stack(x_8, X_8)

# finding P matrix using SVD and last column of V
A = np.vstack((A_1, A_2, A_3, A_4, A_5, A_6, A_7, A_8))
U, D, Vt = np.linalg.svd(A)
V = np.transpose(Vt)
p = V[:, -1]
P = np.reshape(p, (3, 4))

# finding C with SVD and last column of V
U, D, Vt = np.linalg.svd(P)
V = np.transpose(Vt)
C = V[:, -1]


# doing Given Rotations to get the K matrix
M = P[0:3, 0:3]
c = -M[2, 2]/pow(pow(M[2, 1], 2) + pow(M[2, 2], 2), 0.5)
s = M[2, 1]/pow(pow(M[2, 1], 2) + pow(M[2, 2], 2), 0.5)
R_x = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

M = np.dot(M, R_x)
c = -M[2, 2]/pow(pow(M[2, 2], 2) + pow(M[2, 0], 2), 0.5)
s = M[2, 0]/pow(pow(M[2, 2], 2) + pow(M[2, 0], 2), 0.5)
R_y = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])

M = np.dot(M, R_y)
c = -M[1, 1]/pow(pow(M[1, 1], 2) + pow(M[1, 0], 2), 0.5)
s = M[1, 0]/pow(pow(M[1, 1], 2) + pow(M[1, 0], 2), 0.5)
R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

K = np.dot(M, R_z)
print(K)



