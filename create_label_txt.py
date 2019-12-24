"""
create_label_files.py
---------------

This script produces:

"""

import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import glob
import os

ext = ".jpg"
img_path = "/home/khg/Python_proj/ObjectDatasetTools/timer/JPEGImages/"
save_path = "/home/khg/Python_proj/singleshotpose/BRICK/red_brick_2/labels/"

points = []

def on_mouse(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(len(points), 'th pts: ', (x,y))


if __name__ == "__main__":
    folders = [img_path]
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    for classlabel,folder in enumerate(folders):
    
        file_list = os.listdir(img_path)
        file_list_ext = [file for file in file_list if file.endswith(ext)]
        file_list_ext.sort()

        for idx, filename in enumerate(file_list_ext):
            img = cv2.imread(os.path.join(img_path, filename))
            
            h, w, c = img.shape
            cv2.namedWindow(filename)
            end = cv2.setMouseCallback(filename, on_mouse)
            cv2.imshow(filename, img)
            key = cv2.waitKey(0)

            if key == ord("d") and len(points) > 0:
                points.pop()
            
            elif key == ord("n"):
                cv2.destroyAllWindows()
                # if len(points) == 9:
                #     print(os.path.join(save_path, filename[:-4] + ".txt"))
                #     file = open(os.path.join(save_path, filename[:-4] + ".txt"),"w")
                #     message = str(classlabel)[:8] + " "
                #     file.write(message)
                #     x = []
                #     y = []
                #     for point in points:
                #         message = str(point[0]/w)[:8]  + " "
                #         file.write(message)
                #         message = str(point[1]/h)[:8]  + " "
                #         file.write(message)
                #         x.append(point[0])
                #         y.append(point[1])

                #     message = str((max(x)-min(x))/w)[:8]  + " "
                #     file.write(message) 
                #     message = str((max(y)-min(y))/h)[:8]
                #     file.write(message)
                #     file.close()

                if len(points) == 8:
                    print(os.path.join(save_path, filename[:-4] + ".txt"))
                    file = open(os.path.join(save_path, filename[:-4] + ".txt"),"w")
                    message = str(classlabel)[:8] + " "
                    file.write(message)
                    x = []
                    y = []
                    for point in points:
                        x.append(point[0])
                        y.append(point[1])
                    center_x = sum(x)/len(x)
                    center_y = sum(y)/len(y)

                    message = str(center_x/w)[:8]  + " "
                    file.write(message)
                    message = str(center_y/h)[:8]  + " "
                    file.write(message)
                    for point in points:
                        message = str(point[0]/w)[:8]  + " "
                        file.write(message)
                        message = str(point[1]/h)[:8]  + " "
                        file.write(message)

                    message = str((max(x)-min(x))/w)[:8]  + " "
                    file.write(message) 
                    message = str((max(y)-min(y))/h)[:8]
                    file.write(message)
                    file.close()

                points.clear()
                continue

            elif key == ord("c"):
                break
            


