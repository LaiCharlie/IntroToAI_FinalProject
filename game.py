from assets_loader import *
from hand_detect   import *
from classes   import Player
from threading import Thread
from cnn_predict import cnn_predict_image
from teachable   import predict_image
from pynput import mouse
from PIL import Image
from fer import FER  

import pygetwindow as gw
import tensorflow  as tf
import mediapipe   as mp
import tkinter as tk
import numpy as np
import cv2 as cv
import pyautogui
import pygame
import socket
import pickle
import ctypes
import math
import time
import sys

pygame.init()
class Game:
    def __init__(self, screen, music, nickname=None):
        self.screen = screen
        self.music = music
        if nickname is not None:
            self.player = Player(nickname)

        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_id = ''
        self.connected = False
        self.active_conns = 0
        self.thread = None

        # [Lobby name, Lobby password]
        self.lobby_specs   = ['', None]  
        self.lobby_filters = [False, False]  
        self.lobby_picker_arrows_pos = 0
        self.lobby_list = []
        self.lobby_owner_id = ''

        self.pen = Pen()
        self.nick_list = []
        self.chat_list = []
        self.emotion_list = []
        self.rounds_left = 0
        self.game_page = 'In_Lobby_Picker'
        self.winner = None

        self.counter = 60
        self.words = []
        self.painter = ''

        self.pos = (0, 0)
        self.rel = (0, 0)
        self.board_x = 415 * self.screen.getWidthScale(), 1420 * self.screen.getWidthScale()
        self.board_y = 100 * self.screen.getHeightScale(), 925 * self.screen.getHeightScale()
        
        self.current_emotion = ["neutral"]
        self.last_emotion_sent_time = time.time()
        self.face_detected = False
        self.flag = True
        self.predicted_class = None

    def getUniversalPos(self, pos):
        return int(round(pos[0] * 1920) / self.screen.getWidth()), int(round(pos[1] * 1080) / self.screen.getHeight())

    def getUniquePos(self, pos):
        return int(pos[0] * self.screen.getWidthScale()), int(pos[1] * self.screen.getHeightScale())

    def setPlayer(self, nickname):
        self.player = Player(nickname)

    def send(self, data):
        try:
            try:
                msg = pickle.dumps(data)
                self.client.sendall(msg)
            except Exception as e:
                if self.connected:
                    print(f"[ERROR] An error occurred while trying to 'send':\n{e}")
                    if str(e) == '[WinError 10054] An existing connection was forcibly closed by the remote host':
                        self.connected = False
                        self.client.close()
                        self.screen.update()
        except:
            pass

    def recv(self):
        try:
            try:
                msg = pickle.loads(self.client.recv(1024))
                return msg
            except Exception as e:
                if str(e) == '[WinError 10054] An existing connection was forcibly closed by the remote host':
                    self.connected = False
                    self.client.close()
                    self.screen.update()
        except Exception as e:
            if self.connected:
                print(f"[ERROR] An error occurred while trying to 'recieve':\n{e}")

    def connect(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if self.connected is False:
            try:
                self.game_page = 'In_Lobby_Picker'
                lobby_name_box.setText('')
                lobby_password_box.setText('')
                self.client.connect(self.player.getAddr())
                print(f"\n_____________________________________________\n[CONNECTION] You have been connected to the server.\n[NICKNAME] {self.player.getNickname()}")
                data = self.recv()
                if data[0] == 'NICK':
                    self.send(self.player.getNickname())
                    self.client_id = data[1]
                    print(f"[CLIENT-ID] {self.client_id}\n_____________________________________________")
                self.connected = True
            except Exception as e:
                print(f"[ERROR] Connection problem has occurred!: {e}")
                self.connected = False

    def close(self):
        if self.player.getIsPainter() and self.rounds_left > 0 and self.active_conns > 1:
            self.send(['ROUND_OVER', None])
            time.sleep(0.2)
        self.send('DISCONNECT')
        self.client.close()

    def handle_game(self):
        while self.connected:
            try:
                msg = self.recv()
                if type(msg) is Pen:
                    self.pen = msg
                else:
                    if type(msg) is list:
                        info = msg.pop(0)
                        if info in self.client_cmd_dict().keys():
                            self.client_cmd_dict()[info](msg)
                    elif type(msg) is str:
                        if msg in self.client_cmd_dict().keys():
                            self.client_cmd_dict()[msg]()
            except Exception as e:
                print(e)

    def client_cmd_dict(self):
        return {'ANNOUNCE_WINNER': self.cmd_announceWinner,
                'NICKNAME_LIST': self.cmd_nicknameList,
                'LOBBIES_SPECS': self.cmd_lobbySpecs,
                'SET_OWNER_ID' : self.cmd_setOwnerID,
                'NEXT_ROUND': self.cmd_nextRound,
                'START_GAME': self.cmd_startGame,
                'IS_PAINTER': self.cmd_isPainter,
                'COUNTDOWN' : self.cmd_countdown,
                'GAME_PREP' : self.cmd_gamePrep,
                'CHAT_MSG'  : self.cmd_chatMsg,
                'WORDS': self.cmd_words,
            }

    def cmd_announceWinner(self, msg):
        print(f"Winner: {msg[0]}")
        self.winner = str(msg[0])
        self.chat_list = []
        for player in self.nick_list:
            self.nick_list[self.nick_list.index(player)] = (player[0], 0)
        self.send(['SCORE', 0])
        self.game_page = 'In_End_Game'

    def cmd_setOwnerID(self, msg):
        self.lobby_owner_id = msg[0]

    def cmd_nicknameList(self, msg):
        self.nick_list = msg
        self.active_conns = len(self.nick_list)
   
    def cmd_chatMsg(self, msg):
        msg = ' '.join(msg)
        if 'has guessed the word!' in msg:
            self.player.setNumOfGuessers(self.player.getNumOfGuessers() + 1)
        self.chat_list.append(msg)
        print(msg)
        # y_offset = -10
        # if 'emotion' in msg:
        #     #find nickname in msg
        #     i=0
        #     name=''
        #     while(msg[i] != ' '):
        #         name += msg[i]
        #         i += 1
        #     for j in range(len(self.nick_list)):
        #         if(self.nick_list[j][0] == name):
        #             break
        #     print('name: ',name)
        #     print('emotion_list: ',len(self.emotion_list))
        #     print('j: ',j)
        #     diff = 0
        #     for i in range (len(self.emotion_list)):
        #         if(name == self.emotion_list[i]["name"]):
        #             diff = 1
        #             break
        #     if(diff==0):
        #         print('append emotion append emotion append emotion append emotion')
        #         self.emotion_list.append({"name": name, "mood": "neutral"})
        #     else:
        #         if('happy' in msg):
        #             self.emotion_list[j]['mood'] = 'happy'
        #         elif('sad' in msg):
        #             self.emotion_list[j]['mood'] = 'sad'
        #         elif('angry' in msg):
        #             self.emotion_list[j]['mood'] = 'angry'
        #         elif('surprise' in msg):
        #             self.emotion_list[j]['mood'] = 'surprise'
        #         elif('neutral' in msg):
        #             self.emotion_list[j]['mood'] = 'neutral'
        #         else:
        #             print('replace error')


    def cmd_words(self, msg):
        self.words = msg[0]
        self.rounds_left = len(self.words)

    def cmd_nextRound(self, msg):
        global draw_on
        draw_on = False
        del msg[1][0]
        self.nick_list = msg[1]
        self.counter = 100
        self.player.setGuessed(False)
        self.player.setNumOfGuessers(0)
        del self.words[0]
        self.rounds_left = len(self.words)
        self.painter = msg[2]
        self.player.setIsPainter(True)
        self.screen.getScreen().fill((0, 0, 0))
        self.screen.blit(painter_interface, (0, 0))
        
    def cmd_countdown(self, msg):
        self.counter = int(msg[0])

    def cmd_isPainter(self):
        self.player.setIsPainter(True)

    def cmd_startGame(self, msg):
        global draw_on
        self.rounds_left = 6
        draw_on = False
        self.painter = msg[1]
        if self.client_id == msg[0]:
            self.player.setIsPainter(True)
        else:
            self.player.setIsPainter(False)
        self.player.setScore(0)
        self.game_page = 'In_Mid_Game'

    def cmd_gamePrep(self, msg):
        del msg[0][0]
        self.nick_list = msg[0]
        self.active_conns = len(self.nick_list)
        self.words = msg[1]
        self.lobby_owner_id = msg[2]
        self.game_page = 'In_Pre_Game'

    def cmd_lobbySpecs(self, msg):
        if self.game_page == 'In_Lobby_Picker' and self.connected:
            self.lobby_list = msg
        elif not self.connected:
            self.close()

    def game_manager(self):
        global timer_event
        if not self.connected:
            self.connect()
        if self.connected:
            time_delay = 1000
            timer_event = pg.USEREVENT + 1
            pg.time.set_timer(timer_event, time_delay)
        start_thread = True

        pygame.init()
        pygame.display.set_caption("Player")
        screenshot_interval = 5  
        screenshot_count = 0
        last_screenshot_time = time.time()
        window_title = "Player"

        video = cv.VideoCapture(0)
        use_virtual_mouse = False
        if not video.isOpened():
            print("No webcam detected. Using regular mouse control.")
        else:
            print("Webcam detected. Using virtual mouse control.")
            use_virtual_mouse = True
            self.virtual_mouse_thread = Thread(target=self.virtual_mouse_control, args=(video,))
            self.virtual_mouse_thread.start()

        while self.connected:
            try:
                for event in pg.event.get():
                    self.pos, self.rel = pg.mouse.get_pos(), pg.mouse.get_rel()
                    if self.game_page == 'In_Lobby_Picker':
                        self.lobby_picker_gui(event)
                    elif self.game_page == 'In_Pre_Game':
                        self.pre_game_gui(event)
                    elif self.game_page == 'In_Mid_Game':
                        self.mid_game_gui(event)
                        current_time = time.time()

                        if current_time - last_screenshot_time >= screenshot_interval:
                            windows = gw.getWindowsWithTitle(window_title)
                            if windows:
                                win = windows[0] if len(windows) > 1 else windows[1]  
                                win.activate()  
                                if win.width <= 700 and win.height <= 450:
                                    print('a')
                                    window_x = win.left + 160
                                    window_y = win.top  + 88
                                    window_width  = win.width  - 345
                                    window_height = win.height - 156
                                elif win.width <= 850 and win.height <= 550:
                                    print('b')
                                    window_x = win.left + 190
                                    window_y = win.top  + 95
                                    window_width  = win.width  - 462
                                    window_height = win.height - 170
                                elif win.width <= 1050 and win.height <= 650:
                                    print('c')
                                    window_x = win.left + 228
                                    window_y = win.top  + 105
                                    window_width  = win.width  - 500
                                    window_height = win.height - 210
                                elif win.width <= 1350 and win.height <= 800:
                                    print('d')
                                    window_x = win.left + 320
                                    window_y = win.top  + 128
                                    window_width  = win.width  - 680
                                    window_height = win.height - 248
                                elif win.width <= 1650 and win.height <= 1000:
                                    print('e')
                                    window_x = win.left + 390
                                    window_y = win.top  + 150
                                    window_width  = win.width  - 828
                                    window_height = win.height - 300                                    
                                else:
                                    print('f')
                                    window_x = win.left + 228
                                    window_y = win.top + 105
                                    window_width  = win.width - 500
                                    window_height = win.height - 210 

                                screenshot = pyautogui.screenshot(region=(window_x, window_y, window_width, window_height))
                                screenshot_filename = f"./photo/player_{self.words[0]}_screenshot_{screenshot_count}.png"
                                screenshot.save(screenshot_filename)
                                print(f"Screenshot saved: {screenshot_filename}")


                                self.predicted_class, confidence = predict_image(screenshot_filename)
                                def invert_rgb(image):
                                    inverted_image = Image.new("RGB", image.size)
                                    pixels = image.load()
                                    inverted_pixels = inverted_image.load()

                                    for y in range(image.height):
                                        for x in range(image.width):
                                            r, g, b = pixels[x, y]
                                            inverted_pixels[x, y] = (255 - r, 255 - g, 255 - b)

                                    return inverted_image
                                
                                image = Image.open(screenshot_filename).convert("RGB")
                                inverted_image = invert_rgb(image)
                                inverted_filename = f"./photo/inverted_{self.words[0]}_screenshot_{screenshot_count}.png"
                                inverted_image.save(inverted_filename)

                                if self.predicted_class != 'black':
                                    print(f"Predicted Class: {self.predicted_class}")

                                self.predicted_class_png, confidence_png = cnn_predict_image(inverted_filename)
                                print(f"CNN Guess: {self.predicted_class_png}")

                                # with open('guess_log.txt', 'a') as log_file:
                                #     log_file.write(f"Title: {self.words[0]}\n")
                                #     log_file.write(f"[teachable] result {screenshot_filename}: {self.predicted_class} (Confidence: {confidence:.2f})\n")
                                #     log_file.write(f"[self-CNN]  result {screenshot_filename}: {self.predicted_class_png} (Confidence: {confidence_png:.2f})\n")
                                #     log_file.write(f"\n-------------------------------------------------------------\n")

                                screenshot_count += 1
                                last_screenshot_time = current_time    
                                if self.predicted_class == self.words[0] or self.predicted_class_png == self.words[0]:
                                    with open('log/guess_log.txt', 'a') as log_file:
                                        log_file.write(f"Title: {self.words[0]}\n")
                                        log_file.write(f"[teachable] result {screenshot_filename}: {self.predicted_class} (Confidence: {confidence:.2f})\n")
                                        log_file.write(f"[self-CNN]  result {screenshot_filename}: {self.predicted_class_png} (Confidence: {confidence_png:.2f})\n")
                                        log_file.write(f"\n-------------------------------------------------------------\n")

                                    if self.predicted_class == self.words[0]:
                                        self.send(['CHAT_MSG', f'[Teachable] {self.player.getNickname()} has guessed the word!'])
                                    if self.predicted_class_png == self.words[0]:
                                        self.send(['CHAT_MSG', f'[CNN] {self.player.getNickname()} has guessed the word!'])

                                    self.player.setGuessed(True)
                                    chat_input_box.setText('')
                                    self.player.setScore(self.player.getScore() + int(self.counter / (self.player.getNumOfGuessers() + 1)))
                                    self.send(['SCORE', self.player.getScore()])        
                                else :
                                    if self.predicted_class != 'black':
                                        self.send(['CHAT_MSG', f'[Teachable] {self.player.getNickname()}: {self.predicted_class}'])
                                        self.send(['CHAT_MSG', f'[CNN] {self.player.getNickname()}: {self.predicted_class_png}'])

                    elif self.game_page == 'In_End_Game':
                        self.end_game_gui(event)
                    if self.game_page != 'In_Lobby_Picker':
                        self.general_gui(event)
                    self.music.display_icon(self.screen, mute_on, mute_off)
                    self.screen.update()

                if start_thread:
                    self.thread = Thread(target=self.handle_game)
                    self.thread.start()
                    start_thread = False
                    self.send('LOBBIES_SPECS')

            except Exception as e:
                print(e)
                pass

        if use_virtual_mouse:
            self.virtual_mouse_thread.join()
            video.release()
            cv.destroyAllWindows()
            print('end\n')

        self.thread.join()
        self.close()

    def virtual_mouse_control(self, video):
        preTime = 0
        handDetector = HandDetector(minDetectionConfidence=0.7)
        mouseController = mouse.Controller()
        mouseLastX, mouseLastY = -1, -1
        smoothedMouseX, smoothedMouseY = 0, 0

        # Smoothing factor (0 < alpha < 1)
        alpha = 0.2
        screenW, screenH = GetScreenSize()
        print("Screen Size: " + str(screenW) + "x" + str(screenH))
        frameW = int(video.get(3))
        frameH = int(video.get(4))
        print("Camera Frame Resolution:", frameW, frameH)
        vMouseMoveAreaRatio = 2/3
        vMouseRectX, vMouseRectY = frameW / 2, frameH / 2
        vMouseRectW, vMouseRectH = frameW * vMouseMoveAreaRatio, frameH * vMouseMoveAreaRatio
        vMouseRectTopLeftX = int(vMouseRectX - vMouseRectW / 2)
        vMouseRectTopLeftY = int(vMouseRectY - vMouseRectH / 2)
        vMouseRectBtmRightX = int(vMouseRectX + vMouseRectW / 2)
        vMouseRectBtmRightY = int(vMouseRectY + vMouseRectH / 2)
        vMouseRectInfo = (vMouseRectTopLeftX, vMouseRectTopLeftY, vMouseRectBtmRightX, vMouseRectBtmRightY, vMouseRectW, vMouseRectH)
        
        mouseButtonDown = False
        pressStartTime = None 
        longPressThreshold = 1.0  

        last_emotion_detection_time = time.time()

        while self.connected:
            ret, frame = video.read()
            if not ret:
                break
            if videoFlipX:
                frame = cv.flip(frame, 1)
            frame = handDetector.findHands(frame, drawOnImage=True)
            hand0Landmarks = handDetector.findHandPositions(frame, handID=0)
            # hand1Landmarks = handDetector.findHandPositions(frame, handID=1)

            if len(hand0Landmarks) != 0:
                # Middle finger (12) for mouse movement
                middleFingerX, middleFingerY = hand0Landmarks[12][1], hand0Landmarks[12][2]
                cv.circle(frame, (middleFingerX, middleFingerY), 18, (0,120,255), cv.FILLED)

                mouseX, mouseY = FrameXY2ScreenXY(middleFingerX, middleFingerY, vMouseRectInfo, screenW, screenH)
                smoothedMouseX = (1 - alpha) * smoothedMouseX + alpha * mouseX
                smoothedMouseY = (1 - alpha) * smoothedMouseY + alpha * mouseY

                if mouseLastX >= 0:
                    smoothedMouseX, smoothedMouseY = MouseDebounce(smoothedMouseX, smoothedMouseY, mouseLastX, mouseLastY, 10)

                MouseMoveAbs(mouseController, int(smoothedMouseX), int(smoothedMouseY))
                mouseLastX, mouseLastY = int(smoothedMouseX), int(smoothedMouseY)

                cv.putText(frame, "Mouse XY:(" + str(int(smoothedMouseX)) + "," + str(int(smoothedMouseY)) + ")", (middleFingerX, middleFingerY), cv.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
            
                # Index finger (8) for clicking and long pressing
                fingerTipX, fingerTipY = hand0Landmarks[8][1], hand0Landmarks[8][2]
                fingerJointX, fingerJointY = hand0Landmarks[6][1], hand0Landmarks[6][2]
                cv.circle(frame, (fingerTipX, fingerTipY), 18, (0,255,0), cv.FILLED)
                cv.circle(frame, (fingerJointX, fingerJointY), 18, (0,120,255), cv.FILLED)

                if handDetector.isPenHoldingGesture(hand0Landmarks, threshold=30):
                    if not mouseButtonDown:
                        pressStartTime = time.time()
                        mouseController.press(mouse.Button.left)
                        mouseButtonDown = True
                else:
                    if mouseButtonDown:
                        pressEndTime = time.time()
                        pressDuration = pressEndTime - pressStartTime
                        if pressDuration > longPressThreshold:
                            mouseController.click(mouse.Button.left, 2)
                        else:
                            mouseController.release(mouse.Button.left)
                        mouseButtonDown = False
            
            current_time = time.time()

            if current_time - last_emotion_detection_time >= 5:
                last_emotion_detection_time = current_time
                self.face_detected = True
            

            preTime = DisplayFPS(frame, preTime)
            cv.rectangle(frame, (vMouseRectTopLeftX, vMouseRectTopLeftY), 
                        (vMouseRectBtmRightX, vMouseRectBtmRightY), (0, 255, 0), 2, cv.FILLED)
            frame = cv.resize(frame, (videoW, videoH))
            cv.imshow('Virtual Hand Mouse', frame)
            cv.setWindowProperty('Virtual Hand Mouse', cv.WND_PROP_TOPMOST, 1)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    def lobby_picker_gui(self, event):
        self.screen.blit(lobby_picker_background, (0, 0))
        lobby_name_box.run(event, 'SAVE')
        self.checkmark_handler(event)
        tmp_list_len = self.lobby_list_specs_handler(event)

        if 29 * self.screen.getWidthScale() < self.pos[0] < 147 * self.screen.getWidthScale() and 31 * self.screen.getHeightScale() < self.pos[1] < 117 * self.screen.getHeightScale():
            self.music_button(event)
        elif 126 * self.screen.getWidthScale() < self.pos[0] < 436 * self.screen.getWidthScale() and 908 * self.screen.getHeightScale() < self.pos[1] < 990 * self.screen.getHeightScale():
            self.create_lobby_button(event)
        elif 479 * self.screen.getWidthScale() < self.pos[0] < 654 * self.screen.getWidthScale() and 908 * self.screen.getHeightScale() < self.pos[1] < 990 * self.screen.getHeightScale():
            self.join_lobby_button(event)
        elif 1268 * self.screen.getWidthScale() < self.pos[0] < 1292 * self.screen.getWidthScale() and 917 * self.screen.getHeightScale() < self.pos[1] < 939 * self.screen.getHeightScale():
            self.hide_full_lobbies_checkmark(event)
        elif 1268 * self.screen.getWidthScale() < self.pos[0] < 1292 * self.screen.getWidthScale() and 954 * self.screen.getHeightScale() < self.pos[1] < 978 * self.screen.getHeightScale():
            self.hide_locked_lobbies_checkmark(event)
        elif 712 * self.screen.getWidthScale() < self.pos[0] < 735 * self.screen.getWidthScale() and 954 * self.screen.getHeightScale() < self.pos[1] < 977 * self.screen.getHeightScale():
            self.lobby_password_checkmark(event)
        elif 1723 * self.screen.getWidthScale() < self.pos[0] < 1783 * self.screen.getWidthScale() and 886 * self.screen.getHeightScale() < self.pos[1] < 933 * self.screen.getHeightScale():
            self.up_arrow_button(event, tmp_list_len)
        elif 1723 * self.screen.getWidthScale() < self.pos[0] < 1783 * self.screen.getWidthScale() and 962 * self.screen.getHeightScale() < self.pos[1] < 1007 * self.screen.getHeightScale():
            self.down_arrow_button(event, tmp_list_len)
        elif 1538 * self.screen.getWidthScale() < self.pos[0] < 1600 * self.screen.getWidthScale() and 888 * self.screen.getHeightScale() < self.pos[1] < 996 * self.screen.getHeightScale() or \
             1596 * self.screen.getWidthScale() < self.pos[0] < 1645 * self.screen.getWidthScale() and 913 * self.screen.getHeightScale() < self.pos[1] < 971 * self.screen.getHeightScale() or \
             1644 * self.screen.getWidthScale() < self.pos[0] < 1679 * self.screen.getWidthScale() and 936 * self.screen.getHeightScale() < self.pos[1] < 999 * self.screen.getHeightScale():
            self.screen.blit(lobby_picker_back_icon, (0, 0))
            self.back_button(event)
        elif event.type == pg.QUIT:
            self.close()
            time.sleep(2)
            pg.quit()
            sys.exit()
        else:
            hover_sound.reset()
            click_sound.reset()
            error_sound.reset()

    def general_gui(self, event):
        chat_input_box.run(event, 'SEND')
        # self.draw_game_info('')
        self.music.display_icon(self.screen, mute_on, mute_off)
        self.nick_organizer()
        self.chat_organizer()

        if 29 * self.screen.getWidthScale() < self.pos[0] < 147 * self.screen.getWidthScale() and 31 * self.screen.getHeightScale() < self.pos[1] < 117 * self.screen.getHeightScale():
            self.music_button(event)
        elif 71 * self.screen.getWidthScale() < self.pos[0] < 207 * self.screen.getWidthScale() and 32 * self.screen.getHeightScale() < self.pos[1] < 112 * self.screen.getHeightScale() or \
            207 * self.screen.getWidthScale() < self.pos[0] < 270 * self.screen.getWidthScale() and 58 * self.screen.getHeightScale() < self.pos[1] < 89  * self.screen.getHeightScale() or \
            251 * self.screen.getWidthScale() < self.pos[0] < 276 * self.screen.getWidthScale() and 89 * self.screen.getHeightScale() < self.pos[1] < 113 * self.screen.getHeightScale():
            self.screen.blit(back_icon, (0, 0))
            self.back_button(event)
        elif event.type == pg.QUIT:
            self.close()
            time.sleep(2)
            pg.quit()
            sys.exit()
        else:
            hover_sound.reset()
            click_sound.reset()

    def pre_game_gui(self, event):
        if self.active_conns >= 2:
            if self.client_id == self.lobby_owner_id:
                self.screen.blit(start_game_interface, (0, 0))
                if 732 * self.screen.getWidthScale() < self.pos[0] < 1096 * self.screen.getWidthScale() and 936 * self.screen.getHeightScale() < self.pos[1] < 1047 * self.screen.getHeightScale():
                    self.start_button(event)
            else:
                self.screen.blit(waiting_for_owner_to_start_interface, (0, 0))
        else:
            self.screen.blit(waiting_for_players_interface, (0, 0))

    def mid_game_gui(self, event):
        if self.active_conns >= 2 and self.rounds_left > 0:
            self.painter_gui(event)
            self.painter_func(event)
        else:
            time.sleep(0.2)
            self.send('ANNOUNCE_WINNER')

    def end_game_gui(self, event):
        if self.counter > 0:
            winner_sound.play_sound_static()
            self.screen.blit(empty_interface, (0, 0))
            self.screen.draw_text(f'{self.winner.upper()} IS THE WINNER!', (255, 142, 36), 'assets\\fonts\\Dosis-ExtraBold.ttf', 188, 365, 477)
            if event.type == timer_event:
                self.counter -= 12
        else:
            winner_sound.setPlay(True)
            self.counter = 60
            self.screen.blit(empty_interface, (0, 0))
            self.game_page = 'In_Pre_Game'

    def painter_gui(self, event):
        pg.draw.rect(self.screen.getScreen(), self.pen.getColor(), (400 * self.screen.getWidthScale(), 939 * self.screen.getHeightScale(), 200 * self.screen.getWidthScale(), 1030 * self.screen.getHeightScale()))
        self.screen.blit(painter_interface, (0, 0))

        if   1035.5 * self.screen.getWidthScale() < self.pos[0] < 1075.5 * self.screen.getWidthScale() and 970 * self.screen.getHeightScale() < self.pos[1] < 1024 * self.screen.getHeightScale():
            self.clear_button(event)
        elif 1095.5 * self.screen.getWidthScale() < self.pos[0] < 1135.5 * self.screen.getWidthScale() and 970 * self.screen.getHeightScale() < self.pos[1] < 1024 * self.screen.getHeightScale():
            self.thickness_increase(event)
        elif 1155.5 * self.screen.getWidthScale() < self.pos[0] < 1195.5 * self.screen.getWidthScale() and 970 * self.screen.getHeightScale() < self.pos[1] < 1032 * self.screen.getHeightScale():
            self.thickness_decrease(event)
        else:
            pen_sound.reset()

        if self.player.getPenClicked():
            self.pen_func(event)
        elif self.player.getFillClicked():
            self.fill_func(event)
        # elif self.player.getEraserClicked():
        #     self.erase_func(event)

    def guesser_gui(self):
        self.screen.blit(guesser_interface, (0, 0))
        self.draw_countdown(self.counter)
        self.screen.draw_text(str('_ ' * len(self.words[0])).upper(), (255, 255, 255), 'assets\\fonts\\ACETONE.ttf', 100, 780, 42)

    def nick_organizer(self):
        y = 200
        for i in range(len(self.nick_list)):
            size = 190
            size -= 13 * len(self.nick_list[i][0])
            self.screen.draw_text(self.nick_list[i][0], (45, 62, 80), 'assets\\fonts\\Nickname DEMO.otf', size, 40, y)
            if self.game_page == 'In_Mid_Game':
                self.screen.draw_text(f'Score: {self.nick_list[i][1]}', (45, 62, 80), 'assets\\fonts\\Nickname DEMO.otf', 40, 245, y + 80)
            if self.active_conns != len(self.nick_list):
                self.active_conns = len(self.nick_list)
            y += 146

    def chat_organizer(self):
        global y, z
        y = 160
        if len(self.chat_list) <= 20:
            for z in range(len(self.chat_list)):
                self.draw_chat()
        else:
            for z in range(len(self.chat_list) - 20, len(self.chat_list)):
                self.draw_chat()

    def draw_chat(self):
        global y, z
        color = (45, 62, 80)
        if 'has guessed the word!' in self.chat_list[z]:
            color = (79, 232, 19)
        self.screen.draw_text(self.chat_list[z], color, 'assets\\fonts\\Dosis-ExtraBold.ttf', 35, 1460, y)
        y += 38

    def draw_countdown(self, num):
        self.screen.blit(timer_icon, (0, 0))
        self.screen.draw_text(str(num), (255, 255, 255), 'assets\\fonts\\ACETONE.ttf', 70, 461, 42)
    
    def draw_game_info(self, painter):
        if self.rounds_left > 0 and self.game_page == 'In_Mid_Game':
            self.screen.draw_text(f'ROUNDS LEFT: {self.rounds_left}', (255, 255, 255), 'assets\\fonts\\JungleAdventurer.ttf', 55, 1444, 1005)
            self.screen.draw_text(f'PAINTER: {painter}', (255, 255, 255), 'assets\\fonts\\JungleAdventurer.ttf', 55, 1444, 1042)
 
    def painter_func(self, event):
        if self.counter > 0:
            self.send(self.pen)
            self.draw_countdown(self.counter)
            self.screen.draw_text(str(self.words[0]).upper(), (255, 255, 255), 'assets\\fonts\\Dosis-ExtraBold.ttf', 100, 818, 25)

            if event.type == timer_event:
                if self.player.getNumOfGuessers() == len(self.nick_list) - 1:
                    self.send(['COUNTDOWN', 0])
                    self.counter = 0
                else:
                    self.send(['COUNTDOWN', self.counter - 1])
                    self.counter -= 1
        else:
            self.pen = Pen()
            self.send(self.pen)
            self.player.setScore(
                self.player.getScore() + int((self.player.getNumOfGuessers()) * (120 / self.active_conns - 1)))
            self.player.setIsPainter(False)
            time.sleep(0.2)
            self.send(['ROUND_OVER', self.player.getScore()])

        self.board_x = 415 * self.screen.getWidthScale(), 1420 * self.screen.getWidthScale()
        self.board_y = 100 * self.screen.getHeightScale(), 925 * self.screen.getHeightScale()

    def fill_func(self, event):
        self.screen.blit(fill_button_clicked, (0, 0))
        if self.board_x[0] < self.pos[0] < self.board_x[1] and self.board_y[0] < self.pos[1] < self.board_y[1]:
            if event.type == pg.MOUSEBUTTONDOWN:
                self.pen.flood_fill(self.screen, self.pos)
                self.send(['DO_FILL', self.getUniversalPos(self.pos)])

    def pen_func(self, event):
        self.pen.setColor((255, 255, 255))
        self.screen.blit(pen_button_clicked, (0, 0))
        self.draw(event)

    def chat_func(self, event):
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_RETURN:
                if chat_input_box.getText() != '':
                    chat_msg = chat_input_box.getText()
                    if chat_msg.lower() == self.words[0] and not self.player.getGuessed() and not self.player.getIsPainter():
                        self.send(['CHAT_MSG', f'{self.player.getNickname()} has guessed the word!'])
                        self.player.setGuessed(True)
                        chat_input_box.setText('')
                        self.player.setScore(self.player.getScore() + int(self.counter / (self.player.getNumOfGuessers() + 1)))
                    elif chat_msg.lower() != self.words[0]:
                        self.send(['CHAT_MSG', f'{self.player.getNickname()}: {chat_msg}'])
                    elif chat_msg.lower() == self.words[0] and self.player.getGuessed() or self.player.getIsPainter():
                        self.send(['CHAT_MSG', f'{self.player.getNickname()}: {"*" * len(chat_msg)}'])
                    chat_input_box.setText('')

    def music_button(self, event):
        hover_sound.play_sound_static()
        if self.music.getPlay():
            self.screen.blit(mute_on_clicked, (0, 0))
        else:
            self.screen.blit(mute_off_clicked, (0, 0))
        click_sound.play_sound(event)
        self.music.play_music(event)
        
    def clear_button(self, event):
        pen_sound.play_sound_static()
        if event.type == pg.MOUSEBUTTONDOWN:
            self.send('DO_CLEAR')
            click_sound.play_sound(event)
            self.screen.getScreen().fill((0, 0, 0))
            self.screen.blit(painter_interface, (0, 0))
        self.screen.blit(clear_button_clicked, (0, 0))
        
    def back_button(self, event):
        hover_sound.play_sound_static()
        if event.type == pg.MOUSEBUTTONDOWN:
            click_sound.play_sound(event)
            self.connected = False
            self.close()

    def create_lobby_button(self, event):
        hover_sound.play_sound_static()
        self.screen.blit(create_icon_clicked, (0, 0))
        if event.type == pg.MOUSEBUTTONDOWN:
            self.create_lobby_func(event)

    def join_lobby_button(self, event):
        hover_sound.play_sound_static()
        self.screen.blit(join_icon_clicked, (0, 0))
        if event.type == pg.MOUSEBUTTONDOWN:
            self.join_lobby_func(event)

    def up_arrow_button(self, event, tmp_list_len):
        if event.type == pg.MOUSEBUTTONDOWN:
            if tmp_list_len > 11:
                if self.lobby_picker_arrows_pos > 0:
                    self.lobby_picker_arrows_pos -= 1
            else:
                self.lobby_picker_arrows_pos = 0

    def down_arrow_button(self, event, tmp_list_len):
        if event.type == pg.MOUSEBUTTONDOWN:
            if tmp_list_len > 11:
                if self.lobby_picker_arrows_pos < tmp_list_len - 11:
                    self.lobby_picker_arrows_pos += 1
            else:
                self.lobby_picker_arrows_pos = 0

    def start_button(self, event):
        self.screen.blit(owner_start_game_clicked_icon, (0, 0))
        if event.type == pg.MOUSEBUTTONDOWN:
            print(self.pos)
            click_sound.play_sound(event)
            self.send('START_GAME')

    def round_line(self, color, start, end=(0, 0), radius=10):
        if self.player.getIsPainter():
            self.send(['DRAW', self.getUniversalPos(start), self.getUniversalPos(end)])
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = max(abs(dx), abs(dy))
        for i in range(distance):
            x = int(start[0] + float(i) / distance * dx)
            y = int(start[1] + float(i) / distance * dy)
            pg.draw.circle(self.screen.getScreen(), color, (x, y), radius)

    def draw(self, event):
        global draw_on, last_pos
        if self.board_x[0] < self.pos[0] < self.board_x[1] and self.board_y[0] < self.pos[1] < self.board_y[1]:
            if event.type == pg.MOUSEBUTTONDOWN:
                pg.draw.circle(self.screen.getScreen(), self.pen.getColor(), self.pos, self.pen.getPenThickness() * (self.screen.getHeightScale() + self.screen.getWidthScale()) / 2)
                self.send(['DRAW', self.getUniversalPos(self.pos)])
                draw_on = True
            if event.type == pg.MOUSEBUTTONUP:
                draw_on = False
            if event.type == pg.MOUSEMOTION:
                if draw_on:
                    pg.draw.circle(self.screen.getScreen(), self.pen.getColor(), self.pos, self.pen.getPenThickness() * (self.screen.getHeightScale() + self.screen.getWidthScale()) / 2)
                    self.round_line(self.pen.getColor(), self.pos, last_pos, self.pen.getPenThickness() * (self.screen.getHeightScale() + self.screen.getWidthScale()) / 2)
                last_pos = self.pos
        else:
            draw_on = False

    def thickness_increase(self, event):
        pen_sound.play_sound_static()
        self.screen.blit(up_button_clicked, (0, 0))
        if event.type == pg.MOUSEBUTTONDOWN:
            if self.pen.getPenThickness() <= 45:
                click_sound.play_sound(event)
                self.pen.setPenThickness(self.pen.getPenThickness() + 3)

    def thickness_decrease(self, event):
        pen_sound.play_sound_static()
        self.screen.blit(down_button_clicked, (0, 0))
        if event.type == pg.MOUSEBUTTONDOWN:
            if self.pen.getPenThickness() >= 7:
                click_sound.play_sound(event)
                self.pen.setPenThickness(self.pen.getPenThickness() - 3)

    def lobby_password_handle(self, event):
        if self.lobby_specs[1] is None:
            self.screen.blit(password_cover, (0, 0))
        else:
            self.screen.blit(password_checkmark, (0, 0))
            lobby_password_box.run(event, 'SAVE')
            self.lobby_specs[1] = lobby_password_box.getText()

    def hide_full_lobbies_handle(self):
        if self.lobby_filters[0]:
            self.screen.blit(hide_full_lobbies_checkmark, (0, 0))

    def hide_locked_lobbies_handle(self):
        if self.lobby_filters[1]:
            self.screen.blit(hide_locked_lobbies_checkmark, (0, 0))

    def lobby_password_checkmark(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            if self.lobby_specs[1] is None:
                self.lobby_specs[1] = lobby_password_box.getText()
            else:
                self.lobby_specs[1] = None

    def hide_full_lobbies_checkmark(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            if self.lobby_filters[0]:
                self.lobby_filters[0] = False
            else:
                self.lobby_filters[0] = True
            self.lobby_picker_arrows_pos = 0

    def hide_locked_lobbies_checkmark(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            if self.lobby_filters[1]:
                self.lobby_filters[1] = False
            else:
                self.lobby_filters[1] = True
            self.lobby_picker_arrows_pos = 0

    def checkmark_handler(self, event):
        self.lobby_specs[0] = lobby_name_box.getText()
        self.lobby_password_handle(event)
        self.hide_full_lobbies_handle()
        self.hide_locked_lobbies_handle()

    def create_lobby_func(self, event):
        self.send('LOBBIES_SPECS')
        name = ' '.join([x for x in self.lobby_specs[0].split(' ') if len(x) > 0])
        password = self.lobby_specs[1]
        count = 0
        for lobby in self.lobby_list:
            if name == lobby[0][:len(name)]:
                count += 1
        if name != '':
            click_sound.play_sound(event)
            if count > 0:
                name += f' ({count})'
            if password == '':
                password = None
            self.player.setScore(0)
            self.send(['CREATE_LOBBY', name, password])
        else:
            error_sound.play_sound(event)

    def join_lobby_func(self, event):
        name = self.lobby_specs[0]
        password = self.lobby_specs[1]
        if name != '':
            if password == '':
                password = None
            self.player.setScore(0)
            self.send(['JOIN_LOBBY', name, password])
        else:
            error_sound.play_sound(event)

    def draw_lobby_list(self, lobby, color, event):
        global y, z
        self.screen.draw_text(str(lobby[0]), color, 'assets\\fonts\\ChildrenSans.ttf', 80, 130, y)
        self.screen.draw_text(f'Owner: {str(lobby[1]).upper()}', color, 'assets\\fonts\\ChildrenSans.ttf', 80, 1120, y)
        self.screen.draw_text(f'{str(lobby[2])} / 6', color, 'assets\\fonts\\ChildrenSans.ttf', 90, 1674, y)
        if 117 * self.screen.getWidthScale() < self.pos[0] < 1801 * self.screen.getWidthScale() and (y - 2) * self.screen.getHeightScale() < self.pos[1] < (y + 47) * self.screen.getHeightScale():
            if event.type == pg.MOUSEBUTTONDOWN:
                lobby_name_box.setText(str(lobby[0]))
        y += 50
        z += 50

    def filter_lobby_list(self, tmp_list):
        for lobby in self.lobby_list:
            if lobby[4] != 'ACTIVE':
                if self.lobby_filters[0] and not self.lobby_filters[1] and lobby[2] < 6:
                    tmp_list.append(lobby)
                elif not self.lobby_filters[0] and self.lobby_filters[1] and not lobby[3]:
                    tmp_list.append(lobby)
                elif self.lobby_filters[0] and self.lobby_filters[1] and lobby[2] < 6 and not lobby[3]:
                    tmp_list.append(lobby)
                elif not self.lobby_filters[0] and not self.lobby_filters[1]:
                    tmp_list.append(lobby)

                if lobby in tmp_list:
                    if str(lobby_name_box.getText()) == '':
                        pass
                    elif str(lobby_name_box.getText()) != str(lobby[0])[:len(str(lobby_name_box.getText()))]:
                        tmp_list.remove(lobby)
        return tmp_list

    def lobby_list_specs_handler(self, event):
        global y, z
        y = 315
        z = 0
        color = (255, 255, 255)
        tmp_list = []
        tmp_list = self.filter_lobby_list(tmp_list)
        tmp_list_original_len = len(tmp_list)

        if len(tmp_list) > 11:
            tmp_list = tmp_list[self.lobby_picker_arrows_pos:self.lobby_picker_arrows_pos + 11]
        for lobby in tmp_list:
            if lobby is not None:
                if lobby[3] is True:
                    self.screen.blit(lock_icon, (0, 0 + z))
                self.draw_lobby_list(lobby, color, event)
        return tmp_list_original_len
