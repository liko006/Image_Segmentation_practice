import numpy as np
from pycocotools.coco import COCO

# check the data directory
dataDir='/home/JinK/coco/data'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

def get_cat_list(img_id):
    
    # get the annotations ids for given image id
    annIds = coco.getAnnIds(imgIds=[img_id])
    
    # find all categories from the annotations
    cat_ids = []
    for annId in annIds:
        cat_id = coco.loadAnns([annId])[0]['category_id']
        cat_ids.append(cat_id)
    
    # delete the redundant ids
    cat_ids = list(set(cat_ids))
    
    # create one hot vector for representing categories Ids
    cat_one_hot = np.zeros(100, dtype=int)
    for cat_id in cat_ids:
        cat_one_hot[cat_id] = 1
    
    # pair image id and the category vector
    result = {img_id: cat_one_hot}
    
    return result


# example image id = 213255
output = get_cat_list(213255)
print(output)