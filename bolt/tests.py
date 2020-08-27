from django.shortcuts import render
from .forms import FileFieldForm
from web_project import settings
import os
import cv2
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
import json
from sklearn.cluster import KMeans
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as Image_xl
from PIL import Image
import itertools
from django.http import JsonResponse,HttpResponse
from functools import reduce
import shutil
from sklearn.externals import joblib
from bolt.models import feature
from .unet import *
# Create your tests here.
model_path = os.path.join(settings.BASE_DIR,'model')

def phash(img):
    img = img.resize((32,32), Image.ANTIALIAS).convert('L')
    avg = reduce(lambda x, y: x + y, img.getdata()) / 1024.
    hash_value=reduce(lambda x, y: x | (y[1] << y[0]), enumerate(map(lambda i: 0 if i < avg else 1, img.getdata())), 0)
    return hash_value

def hist(img):
    img = img.resize((256,256)).convert("L")
    hist_value = []
    for i in range(0,256,128):
        for j in range(0,256,128):
            sub_image = img.crop((i,j,i+128,j+128)).copy()
            hist_value.append((sub_image.histogram()))
    return hist_value


path = 'D:/web_bolt/ALL_IMG'
imgs_nm=os.listdir(path)
kmeans = joblib.load(os.path.join(model_path,'KMeans_model.m'))

for i,img_nm in enumerate(imgs_nm):
    with Image.open(os.path.join(path,img_nm)) as img:
        p=phash(img)
        h=hist(img)
        a=[[bool(int(d)) for d in str(bin(p))[2:].zfill(1024)]]
        g = kmeans.predict(a)
        feature.objects.create(img_name=img_nm,phash=p,hist=h,group=g[0])
    if i%1000==0:
        print(i)


