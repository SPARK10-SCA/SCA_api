from flask import Flask, Response, request
from flask_cors import CORS
from Model_UNET import Unet
from Model_VGG19 import IntelCnnModel
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import ToTensor
from scipy import ndimage
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
import albumentations as A

import os
import sys
import time
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

INPUT_PATH = '/home/work/hyunbin/sca_api/input/'
OUTPUT_PATH = '/home/work/hyunbin/sca_api/output/'

PART_WEIGHT = '/home/work/hyunbin/sca_api/weights/part/Part.pt'

BREAKAGE_WEIGHT = '/home/work/hyunbin/sca_api/weights/damage/Breakage.pt'
CRUSHED_WEIGHT = '/home/work/hyunbin/sca_api/weights/damage/Crushed.pt'
SCRATCHED_WEIGHT = '/home/work/hyunbin/sca_api/weights/damage/Scratched.pt'
SEPARATED_WEIGHT = '/home/work/hyunbin/sca_api/weights/damage/Separated.pt'

SEVERITY_WEIGHT = '/home/work/hyunbin/sca_api/weights/severity/Severity.pth'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

app = Flask(__name__)
CORS(app)

global origImage

def load_part_unet_model(weight_path):
    model = Unet(encoder="resnet34",pre_weight='imagenet',num_classes=16)
    model = model.to(DEVICE)
    try:
        model.model.load_state_dict(torch.load(weight_path, map_location=torch.device('cuda')))
        return model.model
    except:
        try:
            model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
            return model
        except:
            model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')), strict=False)
            return model

def find_part(predMask):
    part_result = {}
    parts = ["Front bumper","Rear bumper","Front fender(R)","Front fender(L)","Rear fender(R)","Trunk lid","Bonnet","Rear fender(L)","Rear door(R)","Head lights(R)","Head lights(L)","Front Wheel(R)","Front door(R)","Side mirror(R)"]

    # Find and print car part objects
    data_slices = ndimage.find_objects(predMask)
    try:
        if ((damage := data_slices[14]) != None):

            y_start = damage[0].start
            y_stop = damage[0].stop

            x_start = damage[1].start
            x_stop = damage[1].stop

            cut = predMask[y_start:y_stop+1,x_start:x_stop+1,:]

            (values,counts) = np.unique(cut,return_counts=True)
            count_sort_ind = np.argsort(-counts)
            for index in count_sort_ind:
                if values[index] != 0 and values[index] != 15:
                    part_result[parts[values[index]-1]] = round((counts[index] / cut.size) * 100, 1)

            # damage ROI
            damage_ROI = {
                "x_start" : x_start,
                "y_start" : y_start,
                "x_stop" : x_stop,
                "y_stop" : y_stop
            }

            return (part_result, damage_ROI)

    except:
        return None

def make_part_predictions(model, origImage):
    model.eval()
    with torch.no_grad():
    
        tf_toTensor = ToTensor()
        image = tf_toTensor(origImage).float().to(DEVICE)

        predMask = model(image.unsqueeze(0))
        predMask = torch.argmax(predMask, dim=1).detach().cpu().numpy()
        predMask = np.transpose(predMask, (1,2,0))
        
        '''with open(OUTPUT_PATH+'part/predMask.txt', 'w') as outfile:
                for slice_2d in predMask:
                    np.savetxt(outfile, slice_2d)'''

        part_result, damageROI = find_part(predMask)

        return (predMask, part_result, damageROI)

def load_damage_unet_model(weight_path):
    model = Unet(encoder="resnet34",pre_weight='imagenet',num_classes=2)
    model = model.to(DEVICE)
    try:
        model.model.load_state_dict(torch.load(weight_path, map_location=torch.device('cuda')))
        return model.model
    except:
        try:
            model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
            return model
        except:
            model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')), strict=False)
            return model

def find_damage(predMask):
    # Find and print car part objects
    data_slices = ndimage.find_objects(predMask)
    try:
        if ((damage := data_slices[0]) != None):

            y_start = damage[0].start
            y_stop = damage[0].stop

            x_start = damage[1].start
            x_stop = damage[1].stop

            cut = predMask[y_start:y_stop+1,x_start:x_stop+1,:]

            val = round(cut.size/(256*256) * 100, 1)
            if val < 0.1: 
                return None
            else:
                return val 

    except:
        return None

def make_damage_predictions(model1, model2, model3, model4, origImage):
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    with torch.no_grad():
    
        tf_toTensor = ToTensor()
        image = tf_toTensor(origImage).float().to(DEVICE) 

        predMask1 = model1(image.unsqueeze(0))
        predMask1 = torch.argmax(predMask1, dim=1).detach().cpu().numpy()
        predMask1 = np.transpose(predMask1, (1,2,0))

        val1 = find_damage(predMask1)

        predMask2 = model2(image.unsqueeze(0))
        predMask2 = torch.argmax(predMask2, dim=1).detach().cpu().numpy()
        predMask2 = np.transpose(predMask2, (1,2,0))

        val2 = find_damage(predMask2)

        predMask3 = model3(image.unsqueeze(0))
        predMask3 = torch.argmax(predMask3, dim=1).detach().cpu().numpy()
        predMask3 = np.transpose(predMask3, (1,2,0))

        val3 = find_damage(predMask3)

        predMask4 = model4(image.unsqueeze(0))
        predMask4 = torch.argmax(predMask4, dim=1).detach().cpu().numpy()
        predMask4 = np.transpose(predMask4, (1,2,0))

        val4 = find_damage(predMask4)

        return (predMask1, predMask2, predMask3, predMask4, val1, val2, val3, val4)


@app.route('/api/test', methods=['POST'])
def test():
    response = Response()
    if request.method == 'POST':
        print("Enter the index (1-1000): ", end="")
        idx = int(input())
        l = [file for file in os.listdir("/home/work/hyunbin/sca_api/testset/img/")]

        #make original mask
        coco = COCO('../testset/datainfo/testset_info.json')
        img_ids = coco.getImgIds()
        image_id = int(img_ids[idx])
        image_infos = coco.loadImgs(image_id)[0]
        images = cv2.imread(os.path.join('/home/work/hyunbin/sca_api/testset/img/', image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        ann_ids = coco.getAnnIds(imgIds=image_infos['id'])
        anns = coco.loadAnns(ann_ids)
        masks = np.zeros((image_infos["height"], image_infos["width"]))
        for ann in anns:
            pixel_value = ann['category_id'] + 1
            masks = np.maximum(coco.annToMask(ann) * pixel_value, masks)
        resize = A.Compose([A.Resize(width=256, height=256)])
        transformed = resize(image = images, mask=masks)
        masks = transformed["mask"]
        images = transformed["image"]

        if request.files:
            global origImage
            imgData = request.files.get('img')
            origImage = Image.open(imgData)
            origImage = origImage.resize((256, 256))


@app.route('/api/part', methods=['POST'])
def part():
    response = Response()
    model = load_part_unet_model(weight_path=PART_WEIGHT)
    predMask, part_result, damageROI = make_part_predictions(model, origImage)
    if damageROI is not None:
        rect = patches.Rectangle((damageROI[0], damageROI[3]), damageROI[2]-damageROI[0], damageROI[1]-damageROI[3], linewidth=1, edgecolor='r', facecolor='none')
    else: part_result['None'] = True

    result = { part_result, damageROI }
    response.set_data(json.dumps(result, ensure_ascii=False))
    return response

@app.route('/api/damage', methods=['POST'])
def damage():
    response = Response()
    
    model1 = load_damage_unet_model(weight_path=BREAKAGE_WEIGHT)
    model2 = load_damage_unet_model(weight_path=CRUSHED_WEIGHT)
    model3 = load_damage_unet_model(weight_path=SCRATCHED_WEIGHT)
    model4 = load_damage_unet_model(weight_path=SEPARATED_WEIGHT)
    predMask1, predMask2, predMask3, predMask4, val1, val2, val3, val4 = make_damage_predictions(model1, model2, model3, model4, origImage)

    result = {}
    if val1 is None: result['Breakage'] = 0
    else: result['Breakage'] = val1

    if val2 is None: result['Crushed'] = 0
    else: result['Crushed'] = val2

    if val3 is None: result['Scratched'] = 0
    else: result['Scratched'] = val3

    if val4 is None: result['Separated'] = 0
    else: result['Scratched'] = val4

    response.set_data(json.dumps(result, ensure_ascii=False))
    return response


@app.route('/api/severity', methods=['POST'])
def severity():
    model = torch.load("../weights/severity/Severity.pth")

    tf_toTensor = ToTensor()
    image = tf_toTensor(origImage).float().to(DEVICE)
    
    predictions = model(image.unsqueeze(0))
    prediction = predictions[0].detach().cpu()
    severity = np.argmax(prediction)

    figure = plt.subplots(nrows=3, ncols=4, figsize=(10, 10))
    figure.tight_layout()
    figure.savefig('../output/test_output.jpg')

    return severity


if __name__ == '__main__':
    app.run(host="112.175.32.78", port=8443)