import time
import cv2
import torch
import numpy as np

from numpy import random
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
import pywintypes # win32 모듈 오류 발생시 작성할 것(이유는 모르겠는데 해결된다.)
import win32gui, win32ui, win32con, win32api
from multiprocessing import Process, Manager, Value
import matplotlib as mpl
import matplotlib.pylab as plt

import winsound as ws
import multiprocessing as mp
from multiprocessing import Process, Manager, Value


# 영상 화면 조절
ROW = 800
COL = 450

# 화면
x = 675
y = 100
w = 150
h = 150

# 윈도우 화면에서 특정 위치를 잡는 함수
def grab_screen(region=None):
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)


screen = cv2.resize(grab_screen(region=(0, 40, 1280, 745)), (ROW, COL)) # 초기 이미지 선언
image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
img = image_np

# 선언
DEVICE = ''
AUGMENT = False
CONF_THRES = 0.25 # 바운딩 박스 조정
IOU_THRES = 0.45 # 바운딩 박스 조정
CLASSES = None # 분류 필터링 여부
AGNOSTIC_NMS = False # 물체의 바운딩 박스만을 찾고자 할 때
weights = 'best.pt' #이 부분이 학습한 모델을 집어넣는 부 <---- important

device = select_device(DEVICE)
half = device.type != 'cpu'  # half precision only supported on CUDA
print('device:', device)

model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride

names = model.module.names if hasattr(model, 'module') else model.names

if half:
    model.half()  # to FP16

# Main Classification
def detect(SOURCE):

    img_width = int(SOURCE.shape[1])
    IMG_SIZE = 640

    source, imgsz = SOURCE, IMG_SIZE  # 소스, pt파일, 이미지 크기

    ###########################전처리##################################

    # Initialize

    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Get names and colors

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    ##########################끝########################################

    # 이미지 로딩

    img0 = source  # BGR

    assert img0 is not None, 'Image Not Found ' + source

    # Padded resize
    img = letterbox(img0, imgsz, stride=stride)[0]  # 패딩이라는데 모르겠음

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416, pytorch는 채널이 앞에
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t0 = time_synchronized()
    pred = model(img, augment=AUGMENT)[0]  # 핵심 부분인듯 이미지에서 객체를 뽑아냄
    # print('pred shape:', pred.shape)

    # Apply NMS
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)
    # index 0~3은 바운딩 박스의 위치, 4는 확률, 그 외는 나머지 객체일 확률
    # Process detections

    det = pred[0]
    # print('det shape:', det.shape)

    s = ''
    s += '%gx%g ' % img.shape[2:]  # print string
    #############################################수정한 부분#################################################
    
    detected_car = False
    labels = []
    if len(det):
        # Rescale boxes from img_size to img0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        # Print results
        for c in det[:, -1].unique():  # (원래 있던 부분) 검출된 객체를 알려주는 부분으로 보입니다.
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
        # Write results

        for *xyxy, conf, cls in reversed(det):  # 레이어 박스의 위치정보, 정확도, 인식된 객체의 번호
            label = f'{names[int(cls)]} {conf:.2f}'
            if int(xyxy[0]) >= img_width * 0.25 and int(xyxy[2]) <= img_width * 0.75:
                points = label.split(" ")
                if float(points[1]) >= 0.5:
                    labels.append(float(points[1]))
                    detected_car = True
                plot_one_box(xyxy, np.ascontiguousarray(img0), label=label, color=colors[int(cls)], line_thickness=3)

    cv2.imshow("Monitering", img0)  # ROI 영상 출력

    return detected_car, labels

# Option --> Beep 소리나게 하기
def beepsound(toggle, all_stop):
    freq = 2000    # range : 37 ~ 32767
    dur = 1000     # ms

    while 1:
        if all_stop.value == 1:
            break

        if toggle.value == 1:
            ws.Beep(freq, dur) # winsound.Beep(frequency, duration)
            toggle.value = 0

if __name__ == '__main__':
    check_requirements(exclude=('pycocotools', 'thop'))

    all_stop = Value('i', 0)
    toggle = Value('i', 0)

    p1 = Process(target=beepsound, args=(toggle, all_stop,))
    p1.start()

    with torch.no_grad():
        detect(img)

        while 1:

            # region : 윈도우 화면에서 잡고자 하는 위치
            # 영상을 튼 후, 영상 화면을 정확히 맞추어 둘 것
            # (ROW, COL) : 영상 처리에 대한 출력 화면 크기 정의
            window_screen = cv2.resize(grab_screen(region=(0, 40, 1280, 745)), (ROW, COL))
            #cv2.rectangle(img, (x, y), (x + w , y + h), (0, 255, 0))

            capture_image = cv2.cvtColor(window_screen, cv2.COLOR_BGR2RGB)

            detected_car, labels = detect(capture_image) # 영상 출력 및 탐지 객체 클래스 번호 반환함
            toggle.value = detected_car
            if detected_car == True:
                print("!!Police Detected!!")
            ########################################################################################################################

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                all_stop.value = 1
                break




