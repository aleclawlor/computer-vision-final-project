import os 
import cv2
import random 
import numpy as np 
import math 
import pandas as pd 
from string import digits

root_dir = '.\data_copy'
target_dir = '.\data_augmented'

# some of the augmentation techniques and code modeled off of the following source:
# https://towardsdatascience.com/data-augmentation-compilation-with-python-and-opencv-b76b1cd500e0

# randomly rotate an image
def randomly_rotate(image):
    
    rows, cols = image.shape[0], image.shape[1]
    random_angle = random.random() * 360
    rotation_matrix = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows-1)/2.0), random_angle, 1) 
    rotated = cv2.warpAffine(image, rotation_matrix, (cols, rows))

    return rotated

# perform perspective transform on an image
def perspective_transform(image):

    points_1 = np.float32([[50, 65], [365, 50], [30, 390], [390, 390]])
    points_2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    perspective_transform = cv2.getPerspectiveTransform(points_1, points_2)
    transformed = cv2.warpPerspective(image, perspective_transform, (300, 300))

    return transformed


# cutout a random part of the image 
def random_crop(image):

    crop_width = 50
    crop_height = 80 

    # set max x and y coords so we don't go past what's in the image 
    x_max = image.shape[1] - crop_width
    y_max = image.shape[0] - crop_height

    # get random points
    x = random.randint(0, x_max)
    y = random.randint(0, y_max)

    # crop the image
    cropped = image[y: y + crop_height, x: x + crop_width]
    resized = cv2.resize(cropped, (image.shape[1], image.shape[0]))
    return resized


# randomly apply brightness, saturation, or contrast transforms
def color_jitter(image):

    type = random.choice(['saturation', 'brightness', 'contrast'])

    if type == 'saturation':

        value = random.choice([-50, -40, -30, 30, 40, 50])

        #convert to hsv 
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if value > 0:
            limit = 255 - value 

            # apply boolean mask to increase saturation to apply contrast enhancement
            s[s > limit] = 255
            s[s <= limit] += value

        else: 
            limit = np.absolute(value)
            s[s < limit] = 0
            s[s >= limit] -= np.absolute(value)

        hsv = cv2.merge((h, s, v))
        converted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return converted


    elif type == 'brightness':

        value = random.choice([-50, -40, -30, 30, 40, 50])

        #convert to hsv 
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if value > 0:
            limit = 255 - value 

            # apply boolean mask to increase saturation to apply contrast enhancement
            v[v > limit] = 255
            v[v <= limit] += value

        else: 
            limit = np.absolute(value)
            v[v < limit] = 0
            v[v >= limit] -= np.absolute(value)

        hsv = cv2.merge((h, s, v))
        converted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return converted

    elif type == 'contrast':
        brightness = 8
        contrast = random.randint(40, 100)

        img = np.int16(image)
        img = img * (contrast/127+1) - contrast + brightness
        img = np.clip(image, 0, 255)
        img = np.uint8(img)

        return img 
        
# random shear of an image 
def random_shear(image):

    shear_factor = .2
    width, height = image.shape[1], image.shape[0]

    M = np.array([[1, shear_factor, 0], [0, 1, 0]])
    nW = width + (shear_factor * height)
    
    # perform random shear
    sheared = cv2.warpAffine(image, M, (int(nW), height))

    return sheared 


# add noise to an image
def add_gaussian_noise(image):

    mean=0
    var = 10
    sigma = math.sqrt(var)

    # use the normal distribution from numpy 
    gaussian = np.random.normal(mean, sigma, (image.shape[0], image.shape[1]))

    image_with_noise = image.copy()
    image_with_noise[:, :, 0] = image_with_noise[:, :, 0] + gaussian
    image_with_noise[:, :, 1] = image_with_noise[:, :, 1] + gaussian
    image_with_noise[:, :, 2] = image_with_noise[:, :, 2] + gaussian 

    return image_with_noise


df_all = []

# classes used for labelling train and test data 
classes = {
    'glass': 1,
    'paper': 2,
    'cardboard': 3,
    'plastic': 4,
    'metal': 5,
    'trash': 6
}

# loop through all subdirectories
for subdir in next(os.walk(root_dir))[1]:

    curr_path = os.path.join(root_dir, subdir)
    target_path = os.path.join(target_dir, subdir)

    # if the current directory does not exist in the augmented dataset, create it 
    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    # each directory also has another folder within it
    for subdir in next(os.walk(curr_path))[1]:
        path_deeper = os.path.join(curr_path, subdir)
        target_deeper = os.path.join(target_path, subdir)

        if not os.path.isdir(target_deeper):
            os.makedirs(target_deeper)

        # now go through all files and augment
        for file in next(os.walk(path_deeper))[2]:
            file_path = os.path.join(path_deeper, file)
            
            # load the image with opencv 
            image = cv2.imread(file_path)
            image_output_path = os.path.join(target_deeper, file)

            try:

                # with probability of .3, apply a random transform 
                prob = random.random()

                if prob <= .5:
                    transform = random.choice([add_gaussian_noise, random_shear, perspective_transform, random_crop, color_jitter])
                    img_transformed = transform(image)

                    # transformed image output path
                    name_split = file.split('.')
                    transformed_filename = name_split[0] + "_aug." + name_split[1]
                    transformed_path = os.path.join(target_deeper, transformed_filename)

                    cv2.imwrite(transformed_path, img_transformed)
                
                # perform random transforms on the image 
                cv2.imwrite(image_output_path, image)

                material = name_split[0]
                remove_digits = str.maketrans('', '', digits)
                material = material.translate(remove_digits)

                # after writing to image to folder, save in a dataframe to write to Excel
                df_all.append([file, classes[material]])
                df_all.append([transformed_filename, classes[material]])      

            except:
                print("Error writing {} to output file".format(image_output_path))
                continue

df_all = pd.DataFrame(df_all, columns=['image', 'class'])

# shuffle the rows of the dataframe
df_all = df_all.sample(frac=1)

# split the df_all dataframe into validation, train and test splits
# and write these dataframes to csv as well 
# we go with a 60-20-20 split

train, validate, test = np.split(df_all.sample(frac=1, random_state=42),
                                [int(.6*len(df_all)), int(.8*len(df_all))])

df_all.to_csv(os.path.join(target_dir, 'all.csv'), index=False)
train.to_csv(os.path.join(target_dir, 'train.csv'), index=False)
test.to_csv(os.path.join(target_dir, 'test.csv'), index=False)
validate.to_csv(os.path.join(target_dir, 'val.csv'), index=False)