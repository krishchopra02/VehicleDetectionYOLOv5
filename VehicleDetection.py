# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:13:28 2022

@author: krish
"""

import torch
import torchvision
import numpy as np
import cv2
import pafy
from time import time
class VehicleDetection: 
    '''Implements YOLO Model for Vehicle Detection
    using OpenCV'''
    
    def __init__(self,url,model_name):
        self.url = url
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device {self.device}") 
        
    def get_video_from_url(self):
        return cv2.VideoCapture(self.url)
    
    def load_model(self,model_name):
        if model_name:
            model = torch.hub.load('ultralytics/yolov5','custom',path=model_name,force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5','yolov5s',pretrained=True)
        return model
    
    def score_frame(self,frame):
        self.model.to(self.device) 
        frame = [frame]
        results = self.model(frame)
        labels,cord = results.xyxyn[0][:,-1],results.xyxyn[0][:,:-1]
        return labels,cord 
    
    def class_to_label(self,x):
        return self.classes[int(x)]
    
    def plot_boxes(self,results,frame):
        labels,cord =results 
        n = len(labels)
        x_shape,y_shape = frame.shape[1],frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4]>=0.3:
                x1,y1,x2,y2 = int(row[0]*x_shape),int(row[1]*y_shape),int(row[2]*x_shape),int(row[3]*y_shape)
                bgr = (0,0,255)
                cv2.rectangle(frame,(x1,y1),(x2,y2),bgr,2)
                cv2.putText(frame,self.class_to_label(labels[i]),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,1.5,(0,255,0),2)
                
        return frame 
    
    def __call__(self):
        cap = self.get_video_from_url()
        assert cap.isOpened()
        while True:
            ret,frame = cap.read()
            assert ret
            
            frame = cv2.resize(frame,(416,416))
            start_time= time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results,frame)
            
            end_time = time()
            fps = 1/np.round(end_time-start_time,2)
            
            cv2.putText(frame,f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0), 2)
            cv2.imshow('DM PROJECT GROUP 1',frame)
            
            if cv2.waitKey(5) & 0xFF ==27:
                break
        cap.release()
        
detector = VehicleDetection(url="vehicle_moving.mp4",model_name="best.pt")
detector()
        
        
        