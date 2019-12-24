import numpy as np

import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import os

corners = np.zeros((9,3))

# ros, ros_2
# source_id = 0
# corners[1] = np.array([-0.0468507, -0.00223507, 0.764345])
# corners[2] = np.array([0.15927, -0.162761, 0.897777])
# corners[3] = np.array([-0.048554, 0.12045, 0.911136])
# corners[4] = np.array([0.168821, -0.0494106, 1.04706])

# corners[5] = np.array([-0.185617, -0.111081, 0.834262])
# corners[6] = np.array([0.0207694, -0.264655, 0.958311])
# corners[7] = np.array([-0.193425, 0.000171289, 0.982096])

# ros_3, ros_4
# source id = 0
# corners[1] = np.array([-0.0228348, 0.0564787, 0.845408])
# corners[2] = np.array([0.178511, -0.109927, 0.994578])
# corners[3] = np.array([0.00103758, 0.204845, 0.978515])
# corners[4] = np.array([0.193694, 0.0306208, 1.11577])

# corners[5] = np.array([-0.168465, -0.022965, 0.937601])
# corners[6] = np.array([0.0288423, -0.17811, 1.06085])
# corners[7] = np.array([-0.143867, 0.126016, 1.05736])

# ros_3, ros_4
# source id = 287
# corners[1] = np.array([-0.14992, -0.205133, 0.965046]) # [-0.14015, -0.213176, 0.978117]
# corners[2] = np.array([0.0419225, -0.0417562, 0.826962])
# corners[3] = np.array([-0.157309, -0.0882812, 1.11631])
# corners[4] = np.array([0.0561115, 0.0845373, 0.982394])

# corners[5] = np.array([-0.00647471, -0.298717, 1.01473])
# corners[6] = np.array([0.186585, -0.144009, 0.896173])
# corners[8] = np.array([0.202448, -0.0260637, 1.03704])

# ros_5, ros_6
source_id = 45
corners[1] = np.array([-0.231717, -0.108633, 0.842719]) 
corners[2] = np.array([0.0369711, -0.00450506, 0.767285])
corners[3] = np.array([-0.230768, 0.0331188, 0.979571])
corners[4] = np.array([0.0400593, 0.122371, 0.911046])

corners[5] = np.array([-0.152743, -0.228909, 0.939147])
corners[6] = np.array([0.116422, -0.135534, 0.884797])
##### corners[8] = np.array([0.128491, -0.0288786, 1.0393])


# ros_7, ros_8
# source_id = 0
# corners[1] = np.array([-0.149774, 0.00421319, 0.809402]) 
# corners[2] = np.array([0.152606, 0.00359227, 0.83476])
# corners[3] = np.array([-0.151774, 0.109764, 0.970322])
# corners[4] = np.array([0.140976, 0.100073, 0.99575])

# corners[5] = np.array([-0.157726, -0.159032, 0.897245])




corners_idx = []
for i, corner in enumerate(corners):
    if not np.array_equal(corner, np.zeros((3,))):
        corners_idx.append(i)

pair_corners_idx = []
for i in range(1,5):
    if i in corners_idx and 9-i in corners_idx:
        pair_corners_idx.append(i)

# 0.3
for i in range(4):
    st = 2*i+1
    ed = st+1
    if st in corners_idx and ed in corners_idx:
        print("(%d,%d) %f"%(st,ed,np.linalg.norm(corners[st]-corners[ed])))
    else:
        print("(%d,%d) nan"%(st,ed))
        

print('--------------------')

# 0.2
for i in range(0,4,2):
    st1 = 2*i+1
    ed1 = st1+2

    st2 = 2*i+2
    ed2 = st2+2

    if st1 in corners_idx and ed1 in corners_idx:
        print("(%d,%d) %f"%(st1,ed1,np.linalg.norm(corners[st1]-corners[ed1])))
    else:
        print("(%d,%d) nan"%(st1,ed1))
    
    if st2 in corners_idx and ed2 in corners_idx:
        print("(%d,%d) %f"%(st2,ed2,np.linalg.norm(corners[st2]-corners[ed2])))
    else:
        print("(%d,%d) nan"%(st2,ed2))
        
print('--------------------')

# 0.3 0.2 0.3 0.2
# print(np.linalg.norm(cor1-cor2), np.linalg.norm(cor2-cor4), np.linalg.norm(cor3-cor4), np.linalg.norm(cor3-cor1))
# print(np.linalg.norm(cor5-cor6), np.linalg.norm(cor6-cor8), np.linalg.norm(cor7-cor8), np.linalg.norm(cor7-cor5))

# calculate avg using existed pair corners
avg_center = np.zeros((3,))
for i in pair_corners_idx:
    avg_center+=corners[i]
    avg_center+=corners[9-i]

avg_center /= float(2*len(pair_corners_idx))
print(avg_center)


# calculate unknown corner using avg_center
for i in range(1,5):
    if i in pair_corners_idx:
        continue
    
    if not i in corners_idx and not 9-i in corners_idx:
        print("At least one of the pair corners should be known.")
        exit()

    target_idx = i
    if i in corners_idx:
        target_idx = 9-i
    corners[target_idx] = 2.*avg_center - corners[i]

# re-calculate center
avg_center = np.sum(corners[1:], axis=0)/8.
print(avg_center)

print('--------------------')

corners[0] = avg_center
# 0.3
for i in range(4):
    st = 2*i+1
    ed = st+1
    print("(%d,%d) %f"%(st,ed,np.linalg.norm(corners[st]-corners[ed])))

print('--------------------')

# 0.2
for i in range(0,4,2):
    st1 = 2*i+1
    ed1 = st1+2

    st2 = 2*i+2
    ed2 = st2+2

    print("(%d,%d) %f"%(st1,ed1,np.linalg.norm(corners[st1]-corners[ed1])))
    print("(%d,%d) %f"%(st2,ed2,np.linalg.norm(corners[st2]-corners[ed2])))
        
print('--------------------')



# P_mat = np.array([[615.4674072265625, 0.0, 319.95697021484375, 0.0], [0.0, 615.7725219726562, 245.20480346679688, 0.0], [0.0, 0.0, 1.0, 0.0]])
P_mat = np.array([[616.8519287109375, 0.0, 326.568603515625, 0.0],[0.0, 616.97021484375, 239.16815185546875, 0.0], [0.0, 0.0, 1.0, 0.0]])

print(P_mat)
print('--------------------')


edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
tf = np.load('/home/khg/Python_proj/ObjectDatasetTools/BRICK/ros6_191213_221548/trasf_45.npy')
print(tf.shape)

imgs_path = '/home/khg/Python_proj/ObjectDatasetTools/BRICK/ros6_191213_221548/JPEGImages/'
labels_path = "/home/khg/Python_proj/ObjectDatasetTools/BRICK/ros6_191213_221548/labels/"
seg_path = "/home/khg/Python_proj/ObjectDatasetTools/BRICK/ros6_191213_221548/mask/"

folders = [imgs_path]

save = True
visualize = True
start = 0
for classlabel, img_dir_path in enumerate(folders):
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)
    if not os.path.exists(seg_path):
        os.makedirs(seg_path)
    
    file_list = os.listdir(img_dir_path)
    file_list.sort()
    num_files = len(file_list)
    print(img_dir_path, num_files)

    for idx in range(start,num_files):
        img_path = os.path.join(img_dir_path, str(idx)+".jpg")

        color_img = cv2.imread(img_path)
        h, w, c = color_img.shape
        bin_img = np.zeros((h,w,c),dtype=np.uint8)
        concatenated_img = np.zeros((h,2*w,c),dtype=np.uint8)
        # print(tf[idx])
        if np.array_equal(tf[idx], np.zeros((4,4))):
            print("%d skip" % idx)
            continue
        
        out_img = False
        corners_2d = np.zeros((9,2))
        for i, corner in enumerate(corners):
            corner_homo = np.ones((4,))
            corner_homo[:3] = corner.T

            # P = np.dot(P_mat,np.linalg.inv(tf[idx]))
            P = np.dot(P_mat,tf[idx])
            corner_2d = np.dot(P, corner_homo)
            
            # make corner_2d [x,y,1]
            corner_2d = corner_2d/corner_2d[2]
            corners_2d[i] = corner_2d[:2]    
            
            # except all 8 corners are not in images            
            if (corner_2d[0]<0 or corner_2d[0]>=w or corner_2d[1]<0 or corner_2d[1] >= h):
                out_img = True
                continue

            # color_img[corner[1], corner[0],:] = [255,0,0]
            cv2.circle(color_img, (int(corner_2d[0]),int(corner_2d[1])), 2, (255,0,0), -1)
            bin_img[int(corner_2d[1]), int(corner_2d[0]),:] = [255,255,255]

        for edge in edges_corners:
            [x1, x2] = corners_2d[[edge[0] + 1, edge[1] + 1], 0]
            [y1, y2] = corners_2d[[edge[0] + 1, edge[1] + 1], 1]
            cv2.line(color_img, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 3)

        if out_img:
            print("%d skip because some corners are out of the image" % idx)
            continue

        # segmentation
        corners_2d_dim = np.expand_dims(corners_2d, axis=1)
        corners_2d_dim = corners_2d_dim.astype(int)
        hull = cv2.convexHull(corners_2d_dim)
        
        cv2.drawContours(bin_img, [hull], 0, (255, 255, 255), thickness=cv2.FILLED)
        
        save_skip = False
        if visualize:
            # show label image and segmentation image
            concatenated_img[:,:w,:] = color_img
            concatenated_img[:,w:,:] = bin_img

            cv2.imshow(str(idx)+".jpg", concatenated_img)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord("d"):
                save_skip = True
            if key == ord("c"):
                break

        if save:
            if save_skip:
                continue
            cv2.imwrite(os.path.join(seg_path, str(idx)+".png"), bin_img)

            print(os.path.join(labels_path, str(idx) + ".txt"))
            file = open(os.path.join(labels_path, str(idx) + ".txt"),"w")
            message = str(classlabel)[:8] + " "
            file.write(message)
            x = []
            y = []
            for point in corners_2d:
                x.append(point[0])
                y.append(point[1])
                message = str(point[0]/w)[:8]  + " "
                file.write(message)
                message = str(point[1]/h)[:8]  + " "
                file.write(message)
                
            message = str((max(x)-min(x))/w)[:8]  + " "
            file.write(message) 
            message = str((max(y)-min(y))/h)[:8]
            file.write(message)
            file.close()

        







# corners = np.array([[485,102],[403,244],[547,318],[635,173]])

# print(corners.shape)
# print(tf[1])



#!/usr/bin/python
# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# the resulting .ply file can be viewed for example with meshlab
# sudo apt-get install meshlab

# """
# This script reads a registered pair of color and depth images and generates a
# colored 3D point cloud in the PLY format.
# """

# import argparse
# import sys
# import os
# from PIL import Image

# focalLength = 525.0
# centerX = 319.5
# centerY = 239.5
# scalingFactor = 5000.0

# def generate_pointcloud(rgb_file,depth_file,ply_file):
#     """
#     Generate a colored point cloud in PLY format from a color and a depth image.
    
#     Input:
#     rgb_file -- filename of color image
#     depth_file -- filename of depth image
#     ply_file -- filename of ply file
    
#     """
#     rgb = Image.open(rgb_file)
#     depth = Image.open(depth_file)
    
#     if rgb.size != depth.size:
#         raise Exception("Color and depth image do not have the same resolution.")
#     if rgb.mode != "RGB":
#         raise Exception("Color image is not in RGB format")
#     if depth.mode != "I":
#         raise Exception("Depth image is not in intensity format")


#     points = []    
#     for v in range(rgb.size[1]):
#         for u in range(rgb.size[0]):
#             color = rgb.getpixel((u,v))
#             Z = depth.getpixel((u,v)) / scalingFactor
#             if Z==0: continue
#             X = (u - centerX) * Z / focalLength
#             Y = (v - centerY) * Z / focalLength
#             points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))
#     file = open(ply_file,"w")
#     file.write('''ply
# format ascii 1.0
# element vertex %d
# property float x
# property float y
# property float z
# property uchar red
# property uchar green
# property uchar blue
# property uchar alpha
# end_header
# %s
# '''%(len(points),"".join(points)))
#     file.close()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='''
#     This script reads a registered pair of color and depth images and generates a colored 3D point cloud in the
#     PLY format. 
#     ''')
#     parser.add_argument('rgb_file', help='input color image (format: png)')
#     parser.add_argument('depth_file', help='input depth image (format: png)')
#     parser.add_argument('ply_file', help='output PLY file (format: ply)')
#     args = parser.parse_args()

#     generate_pointcloud(args.rgb_file,args.depth_file,args.ply_file)