import mediapipe as mp
import tkinter as tk
import cv2 as cv
import ctypes
import math
import time

from pynput import mouse
from fer import FER  

PROCESS_PER_MONITOR_DPI_AWARE = 1
ctypes.windll.shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)
screenScale = ctypes.windll.shcore.GetScaleFactorForDevice(0)

print("Screen Scaling Factor:", screenScale)
class HandDetector():
    def __init__(self, mode=False, maxNumHands=2, modelComplexity=1, minDetectionConfidence=0.5, minTrackingConfidence=0.5):
        self.mode = mode
        self.maxNumHands = maxNumHands
        self.modelComplexity = modelComplexity
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence
        self.mpHands = mp.solutions.hands
        self.handsDetector = self.mpHands.Hands(self.mode, self.maxNumHands, self.modelComplexity, self.minDetectionConfidence, self.minTrackingConfidence)
        self.mpDrawUtils = mp.solutions.drawing_utils

    def findHands(self, img, drawOnImage=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.handsDetector.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                if drawOnImage:
                    self.mpDrawUtils.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS)
        return img

    def findHandPositions(self, img, handID=0, drawOnImage=True, drawColor=(0,255,0)):
        landmarkList = []
        if self.results.multi_hand_landmarks:
            if handID >= len(self.results.multi_hand_landmarks):
                return landmarkList
            handLandmarks = self.results.multi_hand_landmarks[handID]
            for id, landmark in enumerate(handLandmarks.landmark):
                h, w, c = img.shape
                centerX, centerY = int(landmark.x * w), int(landmark.y * h)
                landmarkList.append([id, centerX, centerY])
                if drawOnImage:
                    cv.circle(img, (centerX, centerY), 8, drawColor)
        return landmarkList

    def isOKGesture(self, handLandmarks, threshold=30):
        thumbTipX, thumbTipY = handLandmarks[4][1], handLandmarks[4][2]
        indexTipX, indexTipY = handLandmarks[8][1], handLandmarks[8][2]

        distance = math.hypot(thumbTipX - indexTipX, thumbTipY - indexTipY)
        return distance < threshold
    
    def isPenHoldingGesture(self, handLandmarks, threshold=30):
        thumbTipX, thumbTipY = handLandmarks[4][1], handLandmarks[4][2]
        indexTipX, indexTipY = handLandmarks[8][1], handLandmarks[8][2]
        middleTipX, middleTipY = handLandmarks[12][1], handLandmarks[12][2]

        distance_thumb_index = math.hypot(thumbTipX - indexTipX, thumbTipY - indexTipY)
        distance_index_middle = math.hypot(indexTipX - middleTipX, indexTipY - middleTipY)

        return distance_thumb_index < threshold and distance_index_middle < threshold


def DisplayFPS(img, preTime):
    curTime = time.time()
    if (curTime - preTime == 0):
        return curTime
    fps = 1 / (curTime - preTime)
    cv.putText(img, "FPS:" + str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
    return curTime

def MouseMoveRel(mouseController, relX, relY):
    mouseController.move(relX, relY)

def MouseMoveAbs(mouseController, x, y):
    mouseController.position = (x, y)

def MouseButtonDown(mouseController, button):
    mouseController.press(button)

def MouseButtonUp(mouseController, button):
    mouseController.release(button)

def GetScreenSize():
    root = tk.Tk()
    screenW = root.winfo_screenwidth()
    screenH = root.winfo_screenheight()
    root.destroy()
    return (screenW, screenH)

def FrameXY2ScreenXY(frameX, frameY, vMouseRectInfo, screenW, screenH):
    (x1, y1, x2, y2, w, h) = vMouseRectInfo
    vMouseX = frameX - x1
    if vMouseX < 0:
        vMouseX = 0
    if vMouseX > w:
        vMouseX = w

    vMouseY = frameY - y1
    if vMouseY < 0:
        vMouseY = 0
    if vMouseY >= h:
        vMouseY = h

    vMouseX = vMouseX / w
    vMouseY = vMouseY / h
    return (vMouseX * screenW, vMouseY * screenH)

def MouseDebounce(curX, curY, lastX, lastY, radius):
    distance = math.hypot(curX - lastX, curY - lastY)
    if distance > radius:
        return (curX, curY)
    else:
        return (lastX, lastY)

videoW = 640
videoH = 480
videoFlipX = True
clickEventThreshold = 15
longPressThreshold = 1.0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
emotion_detector = FER()
current_emotion = "Neutral"

def detect_emotions_and_landmarks(frame, current_emotion):
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    emotions = emotion_detector.detect_emotions(rgb_frame)
    
    if emotions:
        emotion_scores = emotions[0]["emotions"]
        highest_emotion = max(emotion_scores, key=emotion_scores.get)
        
        if emotion_scores[highest_emotion] > 0.3:
            current_emotion[0] = highest_emotion
            text = f"{highest_emotion.capitalize()} detected!"
            if highest_emotion == "happy":
                color = (0, 255, 0)
            elif highest_emotion == "sad":
                color = (0, 0, 255)
            elif highest_emotion == "surprise":
                color = (255, 255, 0)
            elif highest_emotion == "angry":
                color = (0, 0, 255)
            else:  # Neutral
                color = (255, 255, 255)
                
            cv.putText(frame, text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA)
    return frame


def main():
    video = cv.VideoCapture(0)
    preTime = 0
    handDetector = HandDetector(minDetectionConfidence=0.7)
    mouseController = mouse.Controller()
    mouseLastX = -1
    mouseLastY = -1
    screenW, screenH = GetScreenSize()
    print("Screen Size: " + str(screenW) + "x" + str(screenH))

    frameW = int(video.get(3))
    frameH = int(video.get(4))
    vMouseMoveAreaRatio = 2 / 3
    vMouseRectX, vMouseRectY = frameW / 2, frameH / 2
    vMouseRectW, vMouseRectH = frameW * vMouseMoveAreaRatio, frameH * vMouseMoveAreaRatio
    vMouseRectTopLeftX, vMouseRectTopLeftY = int(vMouseRectX - vMouseRectW / 2), int(vMouseRectY - vMouseRectH / 2)
    vMouseRectBtmRightX, vMouseRectBtmRightY = int(vMouseRectX + vMouseRectW / 2), int(vMouseRectY + vMouseRectH / 2)
    vMouseRectInfo = (vMouseRectTopLeftX, vMouseRectTopLeftY, vMouseRectBtmRightX, vMouseRectBtmRightY, vMouseRectW, vMouseRectH)
    mouseButtonDown = False
    pressStartTime = None  # Records the time when pressing starts

    # Track the time of the last emotion detection
    last_emotion_detection_time = time.time()
    emotion_detection_interval = 5  # Interval in seconds
    current_emotion = ["Neutral"]

    while True:
        ret, frame = video.read()
        if not ret:
            break
        if videoFlipX:
            frame = cv.flip(frame, 1)

        # 手部检测和鼠标控制
        frame = handDetector.findHands(frame, drawOnImage=True)
        hand0Landmarks = handDetector.findHandPositions(frame, handID=0)
        
        if len(hand0Landmarks) != 0:
            # 使用中指（12）进行鼠标移动
            middleFingerX, middleFingerY = hand0Landmarks[12][1], hand0Landmarks[12][2]
            cv.circle(frame, (middleFingerX, middleFingerY), 18, (0,120,255), cv.FILLED)

            mouseX, mouseY = FrameXY2ScreenXY(middleFingerX, middleFingerY, vMouseRectInfo, screenW, screenH)
            if mouseLastX >= 0:
                mouseX, mouseY = MouseDebounce(mouseX, mouseY, mouseLastX, mouseLastY, 10)
            MouseMoveAbs(mouseController, mouseX, mouseY)
            mouseLastX = mouseX
            mouseLastY = mouseY

            # 使用握笔手势进行点击
            if handDetector.isPenHoldingGesture(hand0Landmarks, threshold=30):
                if not mouseButtonDown:
                    pressStartTime = time.time()  # 开始长按计时
                    mouseController.press(mouse.Button.left)
                    mouseButtonDown = True
            else:
                if mouseButtonDown:
                    # 长按检测
                    pressEndTime = time.time()
                    pressDuration = pressEndTime - pressStartTime
                    if pressDuration > longPressThreshold:
                        mouseController.click(mouse.Button.left, 2)  # 长按后执行双击
                    else:
                        mouseController.release(mouse.Button.left)
                    mouseButtonDown = False

        current_time = time.time()
        if current_time - last_emotion_detection_time >= emotion_detection_interval:
            frame = detect_emotions_and_landmarks(frame, current_emotion)
            last_emotion_detection_time = current_time
            print(f"Current Emotion: {current_emotion[0]}")

        # Draw virtual mouse move area
        cv.rectangle(frame, (vMouseRectTopLeftX, vMouseRectTopLeftY), (vMouseRectBtmRightX, vMouseRectBtmRightY), (255,0,255), 1)

        preTime = DisplayFPS(frame, preTime)

        cv.imshow("MediaPipe Hands with Mouse Control and Emotion Detection", frame)
        if cv.waitKey(1) == ord('q'):
            break

    video.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()