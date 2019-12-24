import random
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2.aruco as aruco
from open3d import *
import numpy as np
import cv2
import os
import glob
from utils.ply import Ply
from utils.camera import *
from registration import icp, feature_registration, match_ransac, rigid_transform_3D
from tqdm import trange
from pykdtree.kdtree import KDTree
import time
import sys
from config.registrationParameters import *
import json
import png

def marker_registration(source,target):
     cad_src, depth_src = source
     cad_des, depth_des = target
 
     gray_src = cv2.cvtColor(cad_src, cv2.COLOR_RGB2GRAY)
     gray_des = cv2.cvtColor(cad_des, cv2.COLOR_RGB2GRAY)
     aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
     parameters = aruco.DetectorParameters_create()
    
     #lists of ids and the corners beloning to each id
     corners_src, _ids_src, rejectedImgPoints = aruco.detectMarkers(gray_src, aruco_dict, parameters=parameters)
     corners_des, _ids_des, rejectedImgPoints = aruco.detectMarkers(gray_des, aruco_dict, parameters=parameters)
     try:
         ids_src = []
         ids_des = []
         for i in range(len(_ids_src)):
              ids_src.append(_ids_src[i][0])
         for i in range(len(_ids_des)):
              ids_des.append(_ids_des[i][0])
     except:
         return None

     common = [x for x in ids_src if x in ids_des]
  
     if len(common) < 2:
          # too few marker matches, use icp instead
          return None

     
     src_good = []
     dst_good = []
     for i,id in enumerate(ids_des):
          if id in ids_src:
               j = ids_src.index(id)
               for count,corner in enumerate(corners_src[j][0]):
                    feature_3D_src = depth_src[int(corner[1])][int(corner[0])]
                    feature_3D_des = depth_des[int(corners_des[i][0][count][1])][int(corners_des[i][0][count][0])]
                    if feature_3D_src[2]!=0 and feature_3D_des[2]!=0:
                         src_good.append(feature_3D_src)
                         dst_good.append(feature_3D_des)
    
     # get rigid transforms between 2 set of feature points through ransac
     try:
          transform = match_ransac(np.asarray(src_good),np.asarray(dst_good))
          return transform
     except:
          return None


def load_images(path, ID):
    
    """
    Load a color and a depth image by path and image ID 

    """
    global camera_intrinsics
    
    img_file = path + 'JPEGImages/%s.jpg' % (ID*LABEL_INTERVAL)
    cad = cv2.imread(img_file)

    depth_file = path + 'depth/%s.png' % (ID*LABEL_INTERVAL)
    reader = png.Reader(depth_file)
    pngdata = reader.read()
    # depth = np.vstack(map(np.uint16, pngdata[2]))
    depth = np.array(tuple(map(np.uint16, pngdata[2])))
    pointcloud = convert_depth_frame_to_pointcloud(depth, camera_intrinsics)


    return (cad, pointcloud)



if __name__ == "__main__":
    
    source_id = 0

    folders = [sys.argv[1]]
    for path in folders:    
        print(path)

        with open(path+'intrinsics.json', 'r') as f:
             camera_intrinsics = json.load(f)

        Ts = []

        n_pcds = int(len(glob.glob1(path+"JPEGImages","*.jpg"))/LABEL_INTERVAL)

        tf = np.zeros((n_pcds, 4,4))

        for target_id in range(n_pcds):
            color_src, depth_src  = load_images(path, source_id)
            color_dst, depth_dst  = load_images(path, target_id)
            res = marker_registration((color_src, depth_src),
                                    (color_dst, depth_dst))

            if res is not None:
                tf[target_id,:,:] = res
            if (target_id % 100 == 0):
                print("%d/%d"%(target_id, n_pcds))     
        np.save(os.path.join(path, 'trasf_%d.npy'%source_id), tf)
