import numpy as np
from pycocotools.coco import COCO

# check the data directory
dataDir='/home/JinK/coco/data'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

def get_imgs(num_of_imgs):
    
    img_data = {}
    
    # get all category ids 
    cat_ids = coco.getCatIds()
    
    # check whetehr each category has enough images or not
    for cat_id in cat_ids:
        
        # find total number of images for choosen category
        id_num_img = len(coco.getImgIds(catIds=cat_id))
        
        # compare the number of images that exists in the dataset and the number of images you want to get 
        if id_num_img < num_of_imgs:
            pass
        else:
            # find all the image ids for choosen category
            all_img_ids = coco.getImgIds(catIds=cat_id)
            
            # randomly choose the desired number of image ids 
            choosen_ids = np.random.choice(all_img_ids, num_of_imgs, replace=False)
            
            # store category and file names in the data dictionary
            img_data[cat_id] = list(choosen_ids)
    
    return img_data


# choose your number of images
num_imgs = 10

img_data = get_imgs(num_imgs)
print(img_data)
