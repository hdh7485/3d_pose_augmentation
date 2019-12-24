import os
import cv2

augmentation_methods = ['horizontal_flip', 'vertical_flip']

for input_dataset in os.listdir('input_dataset'):
    image_path = 'input_dataset/' + str(input_dataset) + '/JPEGImages'
    image_list = []
    for (dirpath, dirnames, filenames) in os.walk(image_path):
        for file in filenames:
            image_list.append(os.path.join(dirpath, file))
    #print(image_list)

    label_path = 'input_dataset/' + str(input_dataset) + '/labels'
    label_list = []
    for (dirpath, dirnames, filenames) in os.walk(label_path):
        for file in filenames:
            label_list.append(os.path.join(dirpath, file))
    #print(label_list)

    for aug in augmentation_methods:
        print(aug)
        if not os.path.exists('output_dataset/' + str(input_dataset) + '_' + aug + '/JPEGImages'):
            os.makedirs('output_dataset/' + str(input_dataset) + '_' + aug + '/JPEGImages')
        else:
            pass

        if not os.path.exists('output_dataset/' + str(input_dataset) + '_' + aug + '/labels'):
            os.makedirs('output_dataset/' + str(input_dataset) + '_' + aug + '/labels')
        else: 
            pass

        if not os.path.exists('output_dataset/' + str(input_dataset) + '_' + aug + '/mask'):
            os.makedirs('output_dataset/' + str(input_dataset) + '_' + aug + '/mask')
        else: 
            pass

        if aug == 'horizontal_flip':
            for image_path in image_list:
                label_path = 'input_dataset/' + str(input_dataset) + '/labels/' + str(os.path.split(os.path.splitext(image_path)[0])[1]) + '.txt'
                try:
                    f = open(label_path, 'r')
                    line = f.readline()
                    string_values = line.split(' ')
                    for i in range(1, 18, 2):
                        string_values[i] = str(1 - float(string_values[i]))
                    print(string_values)
                    f.close()
                    aug_label_path = 'output_dataset/' + str(input_dataset) + '_' + aug + '/labels/' + str(os.path.split(os.path.splitext(image_path)[0])[1]) + '.txt'
                    with open(aug_label_path, 'w') as f:
                        for value in string_values:
                            f.write("%s " % value)
                except IOError:
                    continue

                mask_path = 'input_dataset/' + str(input_dataset) + '/mask/' + str(os.path.split(os.path.splitext(image_path)[0])[1]) + '.png'
                mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
                flip_horizontal_mask = cv2.flip(mask, 1)
                cv2.imwrite('output_dataset/' + str(input_dataset) + '_' + aug + '/mask/' + str(os.path.split(os.path.splitext(image_path)[0])[1]) + '.png', flip_horizontal_mask)
            
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                flip_horizontal_img = cv2.flip(img, 1)
                cv2.imwrite('output_dataset/' + str(input_dataset) + '_' + aug + '/JPEGImages/' + str(os.path.basename(image_path)), flip_horizontal_img)