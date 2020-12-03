from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import math


from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image


from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import openslide as op
from openslide import OpenSlideError
import PIL
from PIL import Image, ImageDraw, ImageFont
import cv2
import skimage.morphology as sk_morphology



app = Flask(__name__)


#MODEL_PATH = 'model33.h5'
#MODEL_PATH = 'model/saved'
stage_model= 'model/model37.h5'


#model = load_model(MODEL_PATH)
stagemodel=load_model(stage_model)
#model._make_predict_function()   


print('Model loaded. Check http://127.0.0.1:5000/')
def mask_percent(np_img):
  if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
    np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
    mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
  else:
    mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
  return mask_percentage


def filter_grays(rgb, tolerance=15):
    rgb = rgb.astype(np.int)
    rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
    rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
    gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
    result = ~(rg_diff & rb_diff & gb_diff)
    return result

def filter_red(rgb, red_lower_thresh, green_upper_thresh, blue_upper_thresh):
    r = rgb[:, :, 0] > red_lower_thresh
    g = rgb[:, :, 1] < green_upper_thresh
    b = rgb[:, :, 2] < blue_upper_thresh
    result = ~(r & g & b)
    return result

def filter_red_pen(rgb):
    result = filter_red(rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90) & \
               filter_red(rgb, red_lower_thresh=110, green_upper_thresh=20, blue_upper_thresh=30) & \
               filter_red(rgb, red_lower_thresh=185, green_upper_thresh=65, blue_upper_thresh=105) & \
               filter_red(rgb, red_lower_thresh=195, green_upper_thresh=85, blue_upper_thresh=125) & \
               filter_red(rgb, red_lower_thresh=220, green_upper_thresh=115, blue_upper_thresh=145) & \
               filter_red(rgb, red_lower_thresh=125, green_upper_thresh=40, blue_upper_thresh=70) & \
               filter_red(rgb, red_lower_thresh=200, green_upper_thresh=120, blue_upper_thresh=150) & \
               filter_red(rgb, red_lower_thresh=100, green_upper_thresh=50, blue_upper_thresh=65) & \
               filter_red(rgb, red_lower_thresh=85, green_upper_thresh=25, blue_upper_thresh=45)
    return result

def filter_green(rgb, red_upper_thresh, green_lower_thresh, blue_lower_thresh):
    r = rgb[:, :, 0] < red_upper_thresh
    g = rgb[:, :, 1] > green_lower_thresh
    b = rgb[:, :, 2] > blue_lower_thresh
    result = ~(r & g & b)
    return result


def filter_green_pen(rgb):
    result = filter_green(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140) & \
               filter_green(rgb, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110) & \
               filter_green(rgb, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100) & \
               filter_green(rgb, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60) & \
               filter_green(rgb, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210) & \
               filter_green(rgb, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225) & \
               filter_green(rgb, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200) & \
               filter_green(rgb, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20) & \
               filter_green(rgb, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40) & \
               filter_green(rgb, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35) & \
               filter_green(rgb, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60) & \
               filter_green(rgb, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105) & \
               filter_green(rgb, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180) & \
               filter_green(rgb, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150) & \
               filter_green(rgb, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195)
    return result

def filter_blue(rgb, red_upper_thresh, green_upper_thresh, blue_lower_thresh):
    r = rgb[:, :, 0] < red_upper_thresh
    g = rgb[:, :, 1] < green_upper_thresh
    b = rgb[:, :, 2] > blue_lower_thresh
    result = ~(r & g & b)
    return result

def filter_blue_pen(rgb):
    result = filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190) & \
               filter_blue(rgb, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200) & \
               filter_blue(rgb, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230) & \
               filter_blue(rgb, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210) & \
               filter_blue(rgb, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160) & \
               filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130) & \
               filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180) & \
               filter_blue(rgb, red_upper_thresh=40, green_upper_thresh=35, blue_lower_thresh=85) & \
               filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=20, blue_lower_thresh=65) & \
               filter_blue(rgb, red_upper_thresh=90, green_upper_thresh=90, blue_lower_thresh=140) & \
               filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=120) & \
               filter_blue(rgb, red_upper_thresh=110, green_upper_thresh=110, blue_lower_thresh=175)
    return result

def filter_remove_small_objects(np_img, min_size=1000, avoid_overmask=True, overmask_thresh=80):
    rem_sm = sk_morphology.remove_small_objects(np_img, min_size=min_size)
    mask_percentage = mask_percent(rem_sm)
    if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
        new_min_size = min_size / 2
        rem_sm = filter_remove_small_objects(np_img, new_min_size, avoid_overmask, overmask_thresh)
    np_img = rem_sm
    return np_img

def mask_rgb(rgb, mask):
    result = rgb * np.dstack([mask, mask, mask])
    return result

def filter_green_channel(np_img, green_thresh=200, avoid_overmask=True, overmask_thresh=80):
    g = np_img[:, :, 1]
    gr_ch_mask = (g < green_thresh) & (g > 0)
    mask_percentage = mask_percent(gr_ch_mask)
    if (mask_percentage >= overmask_thresh) and (green_thresh < 255) and (avoid_overmask is True):
        new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
        gr_ch_mask = filter_green_channel(np_img, new_green_thresh, avoid_overmask, overmask_thresh)
    np_img = gr_ch_mask
    return np_img

def apply_image_filters(rgb):
    mask_not_green = filter_green_channel(rgb)
    rgb_not_green  = mask_rgb(rgb,mask_not_green) #new
    mask_not_gray = filter_grays(rgb)
    rgb_not_gray  = mask_rgb(rgb,mask_not_gray) #new
    mask_no_red_pen = filter_red_pen(rgb)
    rgb_no_red_pen  = mask_rgb(rgb,mask_no_red_pen) #new
    mask_no_green_pen = filter_green_pen(rgb)
    rgb_no_green_pen  = mask_rgb(rgb,mask_no_green_pen) #new
    mask_no_blue_pen = filter_blue_pen(rgb)
    rgb_no_blue_pen = mask_rgb(rgb,mask_no_blue_pen) #new
    mask_gray_green_pens = mask_not_gray & mask_not_green & mask_no_red_pen & mask_no_green_pen & mask_no_blue_pen
    rgb_gray_green_pens = mask_rgb(rgb,mask_gray_green_pens) #new
    mask_remove_small = filter_remove_small_objects(mask_gray_green_pens, min_size=500)
    rgb_remove_small = mask_rgb(rgb, mask_remove_small)
    not_greenish = filter_green(rgb_remove_small,red_upper_thresh=125,green_lower_thresh=30,blue_lower_thresh=30) #new
    not_grayish = filter_grays(rgb_remove_small, tolerance=30)
    #not_bluish = filter_blue(rgb_remove_small, red_upper_thresh=20, green_upper_thresh=35, blue_lower_thresh=30) #old
    rgb_new = mask_rgb(rgb_remove_small, not_greenish & not_grayish )
    return rgb_new


def preprocess(img_path):
	SCALE_FACTOR = 64
	slide = op.open_slide(img_path)
	large_w, large_h = slide.dimensions
	new_w = math.floor(large_w / SCALE_FACTOR)
	new_h = math.floor(large_h / SCALE_FACTOR)
	level = slide.get_best_level_for_downsample(SCALE_FACTOR)
	whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
	whole_slide_image = whole_slide_image.convert("RGB")
	img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
	img = img.resize((299, 299), PIL.Image.BILINEAR)
	#x = image.img_to_array(img)
	x = np.array(img)
	ias= np.float32(x)
	cv2.imwrite('/home/ridhanya/Desktop/deployment/static/ima.jpg' ,ias) 
	x=apply_image_filters(x)
	


	im=np.float32(x)
	cv2.imwrite('/home/ridhanya/Desktop/deployment/static/im.jpg' ,im) 
	return x



def model_predict(img,stagemodel):


	img= np.expand_dims(img, axis=0)
	
	img=img/255



	#preds = model.predict(img)
	stage=stagemodel.predict(img)
	#print(model.summary())
	#print(preds)
	return stage
    



    

    


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        f = request.files['file']

        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        img=preprocess(file_path)

        

        

        
        stage= model_predict(img, stagemodel)
        print(stage)
        """count=0
        for y in np.nditer(preds):
        	if(y>=0.5 and count ==0):
        		
        		ans="Normal (No Cancer)"
        		count=1
        		break
        	else:
        		ans="Positive (Ovarian Cancer confirmed)"
        		count=1
        		break"""
        st="ERROR"
        cou=0

        for d in np.nditer(stage):
        	cou=cou+1
        	if(d>=0.5 and cou==1):
        		st="Stage 1 Ovarian Cancer"
        		break
        	elif(d>=0.5 and cou==2):
        		st="Stage 2 Ovarian Cancer"
        		break
        	elif(d>=0.5 and cou==3):
        		st="Stage 3 Ovarian Cancer"
        		break
        	elif(d>=0.5 and cou==4):
        		st="Stage 4 Ovarian Cancer"
        		break
        	elif(d>=0.5 and cou==5):
        		st="Normal - Cancer Negative"
        		break


                 
        #pred_class = decode_predictions(preds, top=1)   
                     
        return (st)
    return None

@app.route("/shot", methods=['GET', 'POST'])
def shot():
	return render_template('pred.html',user_image="static/im.jpg",pred_image="static/ima.jpg")





if __name__ == '__main__':
    app.run(debug=True)
