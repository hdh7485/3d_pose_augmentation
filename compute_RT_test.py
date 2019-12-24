import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') 

from registration import icp, feature_registration, match_ransac, rigid_transform_3D
import numpy as np
from scipy.linalg import svd
start_points = np.array([[-0.1,-0.1,0.], [-0.1,-0.1,0.3], [-0.1,0.1,0.], [-0.1,0.1,0.3],
                        [0.1,-0.1,0.], [0.1,-0.1,0.3], [0.1,0.1,0.], [0.1,0.1,0.3]])

end_points = np.array([[-0.15, -0.1, 0.], [0.15, -0.1, 0.], [-0.15, -0.1, 0.2], [0.15, -0.1, 0.2],
                        [-0.15, 0.1, 0.], [0.15, 0.1, 0.], [-0.15, 0.1, 0.2], [0.15, 0.1, 0.2]])



try:
    transform = match_ransac(np.asarray(start_points),np.asarray(end_points))
    print(transform)
except:
    print('no')

m = start_points.shape[0]
transform = np.array(transform)

homo = np.ones((m,4))
homo[:,:3] = start_points
for i in range(m):
    point = homo[i]
    tmp = transform.dot(point.T)
    print(tmp/tmp[3])
print('-----------------------')

# pt_pair_matrix = np.zeros((m*3, 16))

# for i in range(m):
#     start_point = np.ones((1,4))
#     start_point[0,:3] = start_points[i]

#     pt_pair_matrix[3*i, 0:4] = start_point
#     pt_pair_matrix[3*i, 4:12] = np.zeros((1,8))
#     pt_pair_matrix[3*i, 12:16] = -end_points[i,0]*start_point

#     pt_pair_matrix[3*i+1, 0:4] = np.zeros((1,4))
#     pt_pair_matrix[3*i+1, 4:8] = start_point
#     pt_pair_matrix[3*i+1, 8:12] = np.zeros((1,4))
#     pt_pair_matrix[3*i+1, 12:16] = -end_points[i,1]*start_point

#     pt_pair_matrix[3*i+2, 0:8] = np.zeros((1,8))
#     pt_pair_matrix[3*i+2, 8:12] = start_point
#     pt_pair_matrix[3*i+2, 12:16] = -end_points[i,2]*start_point
#     print(pt_pair_matrix)
# U, s, VT = svd(pt_pair_matrix)
# print(VT.shape)

# u, s, vh = np.linalg.svd(pt_pair_matrix, full_matrices=True)

# T = np.linalg.solve(pt_pair_matrix, np.zeros((16,1)))
# np.allclose(np.dot(pt_pair_matrix,T),np.zeros((16,1)))

# print(s)
# T = vh.T[:,0]
# print(T)
# print(vh)


# homo = np.ones((m,4))
# homo[:,:3] = start_points
# for i in range(m):
#     point = homo[i]
#     tmp = T.dot(point.T)
#     print(tmp/tmp[3])