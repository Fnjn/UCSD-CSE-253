#!/usr/bin/python
'''
This file is modified based on 
https://github.com/abhishekrana/DeepFashion
'''


'''
list_category_cloth.txt
50
    category_name  category_type

7   Blazer         1
11  Jacket         1
20  Skirt          2
21  Sweatpants     2
29  Jumpsuit       3


list_category_img.txt
289222
image_name                                                             category_label
img/Sheer_Pleated-Front_Blouse/img_00000001.jpg                        3
img/Sheer_Pleated-Front_Blouse/img_00000002.jpg                        3
img/Sheer_Pleated-Front_Blouse/img_00000003.jpg                        3
img/Sheer_Pleated-Front_Blouse/img_00000004.jpg                        3
img/Sheer_Pleated-Front_Blouse/img_00000005.jpg                        3
img/Sheer_Pleated-Front_Blouse/img_00000006.jpg                        3

'''

### IMPORTS

import os
import numpy as np
import glob
import fnmatch
from random import randint
import PIL
from PIL import Image

import logging
# import colored_traceback
# colored_traceback.add_hook(always=True)
FORMAT = "[%(lineno)4s : %(funcName)-30s ] %(message)s"
#logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

dataset_path='dataset_new'
dataset_train_path=os.path.join(dataset_path, 'train')
dataset_val_path=os.path.join(dataset_path, 'validation')
dataset_test_path=os.path.join(dataset_path, 'test')
#dataset_src_path='fashion_data'
fashion_dataset_path='fashion_data/'
dataset_train_info=os.path.join(dataset_train_path, 'train_info.txt')
dataset_val_info=os.path.join(dataset_val_path, 'val_info.txt')

### GLOBALS


## Shorts               : 14195
# Skirt                : 10794
## Jacket               : 7548
# Top                  : 7270
# Jeans                : 5126
# Joggers              : 3260
# Hoodie               : 2910
# Sweatpants           : 2224
# Coat                 : 1539
# Sweatshorts          : 781
# Capris               : 57

### FUNCTIONS

# Create directory structure
def create_dataset_split_structure():
    # if os.path.exists(dataset_path):
    #     shutil.rmtree(dataset_path)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    if not os.path.exists(dataset_train_path):
        os.makedirs(dataset_train_path)

    if not os.path.exists(dataset_val_path):
        os.makedirs(dataset_val_path)

    if not os.path.exists(dataset_test_path):
        os.makedirs(dataset_test_path)


def get_dataset_split_name(image_path_name, file_ptr):
    for line in file_ptr:
        if image_path_name in line:
            dataset_split_name=line.split()[1]
            # logging.debug('dataset_split_name {}'.format(dataset_split_name))
            return dataset_split_name.strip()


def get_gt_bbox(image_path_name, file_ptr):
    for line in file_ptr:
        if image_path_name in line:
            x1=int(line.split()[1])
            y1=int(line.split()[2])
            x2=int(line.split()[3])
            y2=int(line.split()[4])
            bbox = [x1, y1, x2, y2]
            logging.debug('bbox {}'.format(bbox))
            return bbox


# Get category names list
def get_category_names():
    category_names = []
    with open(fashion_dataset_path + '/Anno/list_category_cloth.txt') as file_list_category_cloth:
        next(file_list_category_cloth)
        next(file_list_category_cloth)
        for line in file_list_category_cloth:
            word=line.strip()[:-1].strip().replace(' ', '_')
            category_names.append(word)
    return category_names


# Create category dir structure
def create_category_structure(category_names):

    for idx,category_name in enumerate(category_names):

#         if category_name not in category_name_generate:
#             logging.debug('Skipping category_names {}'.format(category_name))
#             continue

        logging.debug('category_names {}'.format(category_name))


        # Train
        category_path_name=os.path.join(dataset_train_path, category_name)
        logging.debug('category_path_name {}'.format(category_path_name))
        if not os.path.exists(os.path.join(category_path_name)):
            os.makedirs(category_path_name)

        # Validation
        category_path_name=os.path.join(dataset_val_path, category_name)
        logging.debug('category_path_name {}'.format(category_path_name))
        if not os.path.exists(os.path.join(category_path_name)):
            os.makedirs(category_path_name)

        # Test
        category_path_name=os.path.join(dataset_test_path, category_name)
        logging.debug('category_path_name {}'.format(category_path_name))
        if not os.path.exists(os.path.join(category_path_name)):
            os.makedirs(category_path_name)

def get_attribute_of_interest():
    attr = []
    attr_names = []
    with open(fashion_dataset_path + '/Anno/list_attr_cloth.txt') as file_list_attr_cloth:
        next(file_list_attr_cloth)
        next(file_list_attr_cloth)
        for line in file_list_attr_cloth:
            line = line.split()
            if(line[1] in ['3', '4', '5']):
                attr.append(True)
                attr_names.append(line[0])
                logging.debug('select attr {}'.format(line[0]))
            else:
                attr.append(False)
                logging.debug('unselect attr {}'.format(line[0]))
    return np.array(attr), attr_names



def calculate_bbox_score_and_save_img(index, image_path_name, dataset_image_path, gt_x1, gt_y1, gt_x2, gt_y2):

    logging.debug('dataset_image_path {}'.format(dataset_image_path))
    logging.debug('image_path_name {}'.format(image_path_name))

    image_name = image_path_name.split('/')[-1].split('.')[0]
    logging.debug('image_name {}'.format(image_name))

    img_read = Image.open(image_path_name)
    logging.debug('{} {} {}'.format(img_read.format, img_read.size, img_read.mode))

    # Ground Truthc
    image_save_name = image_path_name.split('/')[-2] + '_' + image_path_name.split('/')[-1].split('.')[0]
    image_save_path = dataset_image_path.rsplit('/', 2)[0]
    image_save_path_name = image_save_path + '/' + str(index) + '.jpg'
    logging.debug('image_save_path_name {}'.format(image_save_path_name))
    #img_crop = img_read.crop((gt_y1, gt_x1, gt_y2, gt_x2))
    img_crop = img_read.crop((gt_x1, gt_y1, gt_x2, gt_y2)).resize([64,64])
    img_crop.save(image_save_path_name)
    logging.debug('img_crop {} {} {}'.format(img_crop.format, img_crop.size, img_crop.mode))


# Generate images from fashon-data into dataset
def generate_dataset_images(category_names):


    count=0
    with open(fashion_dataset_path + '/Anno/list_bbox.txt') as file_list_bbox_ptr:
        with open(fashion_dataset_path + '/Anno/list_category_img.txt') as file_list_category_img:
            with open(fashion_dataset_path + '/Eval/list_eval_partition.txt', 'r') as file_list_eval_ptr:

                next(file_list_category_img)
                next(file_list_category_img)
                idx_crop=1
                for line in file_list_category_img:
                    line = line.split()
                    image_path_name = line[0]
                    logging.debug('image_path_name {}'.format(image_path_name))                                 # img/Tailored_Woven_Blazer/img_00000051.jpg
                    image_name = line[0].split('/')[-1]
                    logging.debug('image_name {}'.format(image_name))                                           # image_name img_00000051.jpg
                    image_full_name = line[0].replace('/', '_')
                    logging.debug('image_full_name {}'.format(image_full_name))                                 # img_Tailored_Woven_Blazer_img_00000051.jpg
                    image_category_index=int(line[1:][0]) - 1
                    logging.debug('image_category_index {}'.format(image_category_index))                       # 2

                    

                    dataset_image_path = ''
                    dataset_split_name = get_dataset_split_name(image_path_name, file_list_eval_ptr)

                    if dataset_split_name == "train":
                        dataset_image_path = os.path.join(dataset_train_path, category_names[image_category_index], image_full_name)
                    elif dataset_split_name == "val":
                        dataset_image_path = os.path.join(dataset_val_path, category_names[image_category_index], image_full_name)
                    elif dataset_split_name == "test":
                        dataset_image_path = os.path.join(dataset_test_path, category_names[image_category_index], image_full_name)
                    else:
                        logging.error('Unknown dataset_split_name {}'.format(dataset_image_path))
                        exit(1)

                    logging.debug('image_category_index {}'.format(image_category_index))
                    logging.debug('category_names {}'.format(category_names[image_category_index]))
                    logging.debug('dataset_image_path {}'.format(dataset_image_path))

                    # Get ground-truth bounding boxes
                    gt_x1, gt_y1, gt_x2, gt_y2 = get_gt_bbox(image_path_name, file_list_bbox_ptr)                              # Origin is top left, x1 is distance from y axis;
                                                                                                                                # x1,y1: top left coordinate of crop; x2,y2: bottom right coordinate of crop
                    logging.debug('Ground bbox:  gt_x1:{} gt_y1:{} gt_x2:{} gt_y2:{}'.format(gt_x1, gt_y1, gt_x2, gt_y2))

                    image_path_name_src = os.path.join(fashion_dataset_path, 'Img', image_path_name)
                    logging.debug('image_path_name_src {}'.format(image_path_name_src))

                    calculate_bbox_score_and_save_img(count, image_path_name_src, dataset_image_path, gt_x1, gt_y1, gt_x2, gt_y2)

                    #TODO: Also cropping in test set. Check if required
                    #shutil.copyfile(os.path.join(fashion_dataset_path, 'Img', image_path_name), dataset_image_path)

                    idx_crop = idx_crop + 1
                    logging.debug('idx_crop {}'.format(idx_crop))


                    count = count+1
                    logging.info('count {} {}'.format(count, dataset_image_path))

def generate_labels(attr_idx):

    with open(fashion_dataset_path + '/Anno/list_attr_img.txt') as file_list_attr_img:
        with open(fashion_dataset_path + '/Eval/list_eval_partition.txt', 'r') as file_list_eval_ptr:
            with open(dataset_train_path + '/labels.txt', 'w') as file_train_label:
                with open(dataset_val_path + '/labels.txt', 'w') as file_val_label:
                    with open(dataset_test_path + '/labels.txt', 'w') as file_test_label:

                        count = 0
                        next(file_list_attr_img)
                        next(file_list_attr_img)
                        for line in file_list_attr_img:
                            line = line.split()
                            image_path_name = line[0]
                            line = np.array(line[1:])
                            line = line[attr_idx]
                            # print(line)
                            one_hot = list(map(lambda x: 1 if x == '1' else 0, line))
                     

                            dataset_split_name = get_dataset_split_name(image_path_name, file_list_eval_ptr)
                            logging.debug('saving labels of {}'.format(image_path_name))
                            if dataset_split_name == "train":
                                file_train_label.write(str(count)+ '.jpg' + ' ' + ' '.join(str(x) for x in one_hot) + '\n')
                            elif dataset_split_name == "val":
                                file_val_label.write(str(count)+ '.jpg' + ' ' + ' '.join(str(x) for x in one_hot) + '\n')
                            elif dataset_split_name == "test":
                                file_test_label.write(str(count)+ '.jpg' + ' ' + ' '.join(str(x) for x in one_hot) + '\n')
                            else:
                                logging.error('Unknown dataset_split_name {}'.format(dataset_image_path))
                                exit(1)
                            
                            count += 1

                
            


if __name__ == '__main__':

    create_dataset_split_structure()
    category_names = get_category_names()
    logging.debug('category_names {}'.format(category_names))
    attribute_idx, attribute_name = get_attribute_of_interest()
    logging.debug('Selected labels {}'.format(attribute_name))
    # generate_dataset_images(category_names)
    generate_labels(attribute_idx)




