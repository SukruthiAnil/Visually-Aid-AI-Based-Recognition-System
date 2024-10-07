import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import cv2
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint,LearningRateScheduler
from IPython.display import display
from tensorflow.python.keras import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras import backend
from tensorflow.keras import optimizers
import pyttsx3
import time
import face_recognition
import imutils
import argparse
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadWebcam
from utils.general import check_img_size, check_requirements, check_imshow,non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging,increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from PIL import Image, ImageOps
import easyocr
from tkinter import *

def speech(result):
    engine=pyttsx3.init()
    with open('face.txt',mode ='w') as file:
        file.write(result)
        print(result)
    fh=open("face.txt","r")
    myText = fh.read().replace("\n"," ")
    language = 'en'
    engine.say(myText)
    engine.runAndWait()


def face_rec():
    cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    faceCascade = cv2.CascadeClassifier(cascPathface)
    data = pickle.loads(open('face_enc', "rb").read())
    print("Streaming started")
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(60, 60),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)
        names = []
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    #Check the names at respective indexes we stored in matchedIdxs
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
            names.append(name)
            for ((x, y, w, h), name) in zip(faces, names):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)
                speech(name)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def o_d():
    def detect(opt):
        source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        save_img= not opt.nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        names = model.module.names if hasattr(model, 'module') else model.names # get class names
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))) # run once
        # # Get names and colors
        # names = model.module.names if hasattr(model, 'module') else model.names
        # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    #p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                    p, s, im0, frame = path[i], '', im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                    # p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if opt.save_crop else im0 # for opt.save_crop
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or opt.save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True),line_thickness=opt.line_thickness)
                            if opt.save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')
                speech(s)

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond
                
                print(f'Done.({time.time() - t0:.3f}s)')

                # # Save results (image with detections)
                # if save_img:
                #     if dataset.mode == 'image':
                #         cv2.imwrite(save_path, im0)
                #     else:  # 'video' or 'stream'
                #         if vid_path[i] != save_path:  # new video
                #             vid_path[i] = save_path
                #             if isinstance(vid_writer[i], cv2.VideoWriter):
                #                 vid_writer[i].release()  # release previous video writer
                #             if vid_cap:  # video
                #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
                #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if __name__ == '__main__':
            parser = argparse.ArgumentParser()
            parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
            parser.add_argument('--source', type=str, default='0', help='source') # file/folder, 0 for webcam
            parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
            parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
            parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
            parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
            parser.add_argument('--view-img', action='store_true', help='display results')
            parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
            parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
            parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
            parser.add_argument('--nosave', default=True, action='store_true', help='do not save images/videos')
            parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
            parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
            parser.add_argument('--augment', action='store_true', help='augmented inference')
            parser.add_argument('--update', action='store_true', help='update all models')
            parser.add_argument('--project', default='runs/detect', help='save results to project/name')
            parser.add_argument('--name', default='exp', help='save results to project/name')
            parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
            parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
            parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
            parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
            opt = parser.parse_args()
            print(opt)
            check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

            with torch.no_grad():    
                if opt.update:  # update all models (to fix SourceChangeWarning)
                    for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                        detect()
                        strip_optimizer(opt.weights)
                else:
                    detect(opt=opt)

def emo_rec():
    label_to_text = {0:'Anger', 1:'Disgust',2:'Fear', 3:'Happy', 4:'Sad', 5: 'Surprise', 6:'Neutral'}
    face_haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    model_1 = load_model('FacialKeyPoints_weights.hdf5')
    model_2 = load_model('FacialExpression_weights.hdf5')

    def predict(X_test):
        df_predict = model_1.predict(X_test)

        # Making prediction from the emotion model
        df_emotion = np.argmax(model_2.predict(X_test), axis=-1)

        # Reshaping array from (856,) to (856,1)
        df_emotion = np.expand_dims(df_emotion, axis = 1)
        # Converting the predictions into a dataframe
        df_predict = pd.DataFrame(df_emotion, columns = ['A'])
        return df_predict
    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN

    rectangle_bgr = (255,255,255)
    img = np.zeros((500,500)) #makes a black image
    text = "Some text in a box!"
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale = font_scale, thickness=1)[0]
    text_offset_x = 10
    text_offset_y = img.shape[0] - 25
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
    cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0,0,0), thickness=1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(gray, 1.1, 4)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            # roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
            roi_color = frame[y:y+h, x:x+w]
            face = face_haar_cascade.detectMultiScale(roi_gray)

            for (ex,ey,ew,eh) in face:   
                face_roi = roi_color[ey:ey+eh, ex:ex+ew]
                final_image = cv2.resize(face_roi, (96,96))
                final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
                final_image = np.expand_dims(final_image,axis=0)
                final_image = final_image/255.0 

                font = cv2.FONT_HERSHEY_SIMPLEX

                df_predict = predict(final_image)
                emotion_prediction = df_predict.at[0,'A']

                if emotion_prediction == 0:
                    emo='Angry'
                    x1,y1,w1,h1 = 0,0,175,75
                    cv2.rectangle(frame, (x1,x1), (x1+w1,y1+h1), (0,0,0), -1)
                    cv2.putText(frame, emo, (x1+int(w1/10), y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                    cv2.putText(frame, emo, (100,150), font, 3, (0,0,255), 2, cv2.LINE_4)
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))
                    speech(emo)

                elif emotion_prediction == 1:
                    emo='Disgust'
                    x1,y1,w1,h1 = 0,0,175,75
                    cv2.rectangle(frame, (x1,x1), (x1+w1,y1+h1), (0,0,0), -1)
                    cv2.putText(frame, emo, (x1+int(w1/10), y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.putText(frame, emo, (100,150), font, 3, (0,0,255), 2, cv2.LINE_4)
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))
                    speech(emo)

                elif emotion_prediction == 2:
                    emo='Fear'
                    x1,y1,w1,h1 = 0,0,175,75
                    cv2.rectangle(frame, (x1,x1), (x1+w1,y1+h1), (0,0,0), -1)
                    cv2.putText(frame, emo, (x1+int(w1/10), y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.putText(frame, emo, (100,150), font, 3, (0,0,255), 2, cv2.LINE_4)
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))
                    speech(emo)

                elif emotion_prediction == 3:
                    emo='Happy'
                    x1,y1,w1,h1 = 0,0,175,75
                    cv2.rectangle(frame, (x1,x1), (x1+w1,y1+h1), (0,0,0), -1)
                    cv2.putText(frame, emo, (x1+int(w1/10), y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.putText(frame, emo, (100,150), font, 3, (0,0,255), 2, cv2.LINE_4)
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))
                    speech(emo)

                elif emotion_prediction == 4:
                    emo='Sad'
                    x1,y1,w1,h1 = 0,0,175,75
                    cv2.rectangle(frame, (x1,x1), (x1+w1,y1+h1), (0,0,0), -1)
                    cv2.putText(frame, emo, (x1+int(w1/10), y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.putText(frame, emo, (100,150), font, 3, (0,0,255), 2, cv2.LINE_4)
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))
                    speech(emo)     

                elif emotion_prediction == 5:
                    emo='Surprised'
                    x1,y1,w1,h1 = 0,0,175,75
                    cv2.rectangle(frame, (x1,x1), (x1+w1,y1+h1), (0,0,0), -1)
                    cv2.putText(frame, emo, (x1+int(w1/10), y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.putText(frame, emo, (100,150), font, 3, (0,0,255), 2, cv2.LINE_4)
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))
                    speech(emo)

                else:
                    emo='Neutral'
                    x1,y1,w1,h1 = 0,0,175,75
                    cv2.rectangle(frame, (x1,x1), (x1+w1,y1+h1), (0,0,0), -1)
                    cv2.putText(frame, emo, (x1+int(w1/10), y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)  
                    cv2.putText(frame, emo, (100,150), font, 3, (0,0,255), 2, cv2.LINE_4)
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))
                    speech(emo)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1)== ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def char_rec():
    videoCaptureObject = cv2.VideoCapture(0)
    result = True
    while(result):
        ret,frame = videoCaptureObject.read()
        time.sleep(5)
        cv2.imwrite("opencv_frame.jpg",frame)
        result = False
    videoCaptureObject.release()
    cv2.destroyAllWindows()

    reader=easyocr.Reader(['en'])
    data=reader.readtext('opencv_frame.jpg')
    s=''

    for text in data:
        s+=text[-2]
        s=s+''

    #print(s)
    speech(s)

window = Tk()

b1 = Button(window,text="Character Recognition",width=30,command=char_rec)
b1.grid(row=0,column=0)

b2 = Button(window,text="Object Detection",width=30,command=o_d)
b2.grid(row=0,column=1)

b3 = Button(window,text="Expression Recognition",width=30,command=emo_rec)
b3.grid(row=1,column=0)

b4 = Button(window,text="Face Recognition",width=30,command=face_rec)
b4.grid(row=1,column=1)
window.mainloop()
