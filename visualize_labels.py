import numpy as np

import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import os


edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]

folders = [sys.argv[1]]

save = True
visual = False

for classlabel, folder in enumerate(folders):
    img_dir_path = os.path.join(folder, 'JPEGImages')
    labels_path = os.path.join(folder, 'labels')
    # seg_path = os.path.join(folder, 'mask')

    save_path = os.path.join(folder, 'draw_gt')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_list = os.listdir(labels_path)
    file_list.sort(key=lambda f: int(filter(str.isdigit, f)))
    
    num_files = len(file_list)
    print(folder, num_files)
    

    for idx, label_file in enumerate(file_list):
        filename = label_file[:-4]

        img_path = os.path.join(img_dir_path, filename+".jpg")
        # mask_path = os.path.join(seg_path, filename+".png")
        label_path = os.path.join(labels_path, label_file)

        color_img = cv2.imread(img_path)
        h, w, c = color_img.shape
        # bin_img = cv2.imread(mask_path)

        concatenated_img = np.zeros((h,2*w,c),dtype=np.uint8)

        f = open(label_path, 'r')
        lines=f.readlines()
        f.close()

        for i in range(len(lines)):
            line = lines[i].split('\n')[0]
            
            corners_2d = np.zeros((9,2))
            words = line.split(' ')

            clas = int(words[0])
            for j in range(9):

                corners_2d[j] = np.array([int(float(words[2*j+1])*w), int(float(words[2*j+2])*h)])
            
            for edge in edges_corners:
                [x1, x2] = corners_2d[[edge[0] + 1, edge[1] + 1], 0]
                [y1, y2] = corners_2d[[edge[0] + 1, edge[1] + 1], 1]
                cv2.line(color_img, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 3)


        # show label image and segmentation image
        concatenated_img[:,:w,:] = color_img
        # concatenated_img[:,w:,:] = bin_img

        if save:
            cv2.imwrite(os.path.join(save_path, filename+".jpg"), color_img)

        if visual:
            cv2.imshow("%d/%d  "%(idx, num_files)+filename+".jpg", concatenated_img)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord("d"):
                print(filename)
            if key == ord("c"):
                break
