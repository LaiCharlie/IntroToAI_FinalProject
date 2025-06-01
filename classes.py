from random import sample
from pygame import mixer
import pygame as pg
import cv2
import os
import uuid

class Player(object):
    def __init__(self, nickname):
        self.IP = '127.0.0.1'
        self.PORT = 12356
        self.nickname = nickname
        self.addr = (self.IP, self.PORT)
        self.score = 0
        self.emotion = 'neutral'

        self.fill_clicked = False
        self.eraser_clicked = False
        self.pen_clicked = True

        self.isPainter = True
        self.guessed = False
        self.num_of_guessers = 0

    def getAddr(self):
        return self.addr

    def getNickname(self):
        return self.nickname

    def getIsPainter(self):
        return self.isPainter

    def getGuessed(self):
        return self.guessed

    def getNumOfGuessers(self):
        print(self.num_of_guessers)
        return self.num_of_guessers

    def getPenClicked(self):
        return self.pen_clicked

    def getEraserClicked(self):
        return self.eraser_clicked

    def getFillClicked(self):
        return self.fill_clicked

    def getScore(self):
        return self.score
    
    def getEmotion(self):
        return self.emotion

    def setNickname(self, nickname):
        self.nickname = nickname

    def setIsPainter(self, IsPainter):
        self.isPainter = IsPainter

    def setGuessed(self, guessed):
        self.guessed = guessed

    def setNumOfGuessers(self, num_of_guessers):
        self.num_of_guessers = num_of_guessers

    def setPenClicked(self, pen_clicked):
        self.eraser_clicked = False
        self.fill_clicked = False
        self.pen_clicked = pen_clicked

    def setEraserClicked(self, eraser_clicked):
        self.pen_clicked = False
        self.fill_clicked = False
        self.eraser_clicked = eraser_clicked

    def setFillClicked(self, fill_clicked):
        self.pen_clicked = False
        self.eraser_clicked = False
        self.fill_clicked = fill_clicked

    def setScore(self, score):
        self.score = score

    def setEmotion(self, emotion):
        self.emotion = emotion 

class PgSetup(object):
    def __init__(self, WIDTH, HEIGHT):
        os.environ['SDL_VIDEO_CENTERED'] = '1'
        pg.init()
        pg.key.set_repeat(150, 350)
        pg.font.init()
        pg.display.set_caption('DrawGuess_Client')
        # Windows only
        pg.display.set_icon(pg.image.load('assets\\images\\icons\\programicon.png'))
        self.clock = pg.time.Clock()

        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        self.resolution_options = ['640 x 360', '768 x 432', '960 x 540', '1280 x 720', '1600 x 900', '1920 x 1080']
        if f'{WIDTH} x {HEIGHT}' not in self.resolution_options:
            self.WIDTH = 960
            self.HEIGHT = 540
        else:
            self.WIDTH = WIDTH
            self.HEIGHT = HEIGHT

        self.IDEAL_WIDTH  = 1920
        self.IDEAL_HEIGHT = 1080
        self.width_scale  = self.WIDTH  / self.IDEAL_WIDTH
        self.height_scale = self.HEIGHT / self.IDEAL_HEIGHT

    def getScreen(self):
        return self.screen

    def getWidthScale(self):
        return self.width_scale

    def getHeightScale(self):
        return self.height_scale

    def getClock(self):
        return self.clock

    def getWidth(self):
        return self.WIDTH

    def getHeight(self):
        return self.HEIGHT

    def getResolutionOptions(self):
        return self.resolution_options

    def setWidth(self, width):
        self.WIDTH = width

    def setHeight(self, height):
        self.HEIGHT = height

    def setScreen(self, width, height):
        self.screen = pg.display.set_mode((width, height))

    def update(self, x=None):
        self.clock.tick(60)
        if x is None:
            pg.display.update()
        else:
            pg.display.update(x)

    def flip(self, tick=60):
        pg.display.flip()
        self.clock.tick(tick)

    def blit(self, pic, pos):
        pic = pg.transform.scale(pic, (self.WIDTH, self.HEIGHT))
        self.screen.blit(pic, (pos[0] * self.width_scale, pos[1] * self.height_scale))

    def screen_resize(self, width, height):
        self.WIDTH = width
        self.HEIGHT = height
        self.width_scale = width / self.IDEAL_WIDTH
        self.height_scale = height / self.IDEAL_HEIGHT
        self.screen = pg.display.set_mode((self.WIDTH, self.HEIGHT))
        os.environ['SDL_VIDEO_CENTERED'] = '1'

    def draw_text(self, text, color, font, size, x, y):
        font1 = pg.font.Font(font, int(size * (self.height_scale + self.width_scale / 3) / 2))
        surface = font1.render(text, True, color)
        self.screen.blit(surface, (x * self.width_scale, y * self.height_scale))

    @staticmethod
    def loadify(image):
        return pg.image.load(image).convert_alpha()

    @staticmethod
    def quit():
        pg.display.quit()
        pg.font.quit()

class Sound(object):
    def __init__(self, volume, sound, play=True):
        self.play = play
        self.sound = mixer.Sound(sound)
        self.volume = volume

        if type(self.volume) == 'int' or 'float':
            self.sound.set_volume(volume)
        else:
            self.sound.set_volume(0.2)

    def getVolume(self):
        return self.volume

    def getPlay(self):
        return self.play

    def setVolume(self, volume):
        self.volume = volume

    def setPlay(self, play):
        self.play = play

    def play_sound(self, event):
        if self.play is True:
            if event.type == pg.MOUSEBUTTONDOWN:
                self.sound.play()
                self.play = False
        if event.type == pg.MOUSEBUTTONUP:
            self.play = True

    def play_sound_static(self):
        if self.play is True:
            self.sound.play()
            self.play = False

    def reset(self):
        if self.play is False:
            self.setPlay(True)

class Music:
    # Windows only
    def __init__(self, volume, play=True, music='assets\\sounds\\Theme_Music.mp3'):
        mixer.music.load(music)
        pg.mixer.music.play(-1)
        self.play = play
        self.volume = volume

        if type(self.volume) == 'int' or 'float':
            pg.mixer.music.set_volume(self.volume)
        else:
            pg.mixer.music.set_volume(0.01)

    def getVolume(self):
        return self.volume

    def getPlay(self):
        return self.play

    def setVolume(self, volume):
        if type(self.volume) == 'int' or 'float':
            pg.mixer.music.set_volume(volume)
        else:
            pg.mixer.music.set_volume(0.01)

    def play_music(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            if self.play is False:
                pg.mixer.music.unpause()
                self.play = True
            else:
                pg.mixer.music.pause()
                self.play = False

    def display_icon(self, screen, icon_mute_on, icon_mute_off):
        if self.play is False:
            screen.blit(icon_mute_on, (0, 0))
        else:
            screen.blit(icon_mute_off, (0, 0))

class InputBox(object):
    def __init__(self, screen, x, y, w, h, font, text_size, active_color, inactive_color, max_str_len=10, border=True, text=''):
        self.screen = screen
        self.w = w
        self.h = h
        self.x = x
        self.y = y

        self.color = active_color
        self.inactive_color = inactive_color
        self.active_color = active_color
        self.border = border
        self.active = False

        self.text = text
        self.text_size = text_size
        self.font_raw = font
        self.max_str_len = max_str_len
        self.font = pg.font.Font(self.font_raw, int(self.text_size * self.screen.getHeightScale()))
        self.txt_surface = self.font.render(self.text, True, self.color)
        self.rect = pg.Rect(self.x * self.screen.getWidthScale(), self.y * self.screen.getHeightScale(), self.w, self.h * self.screen.getHeightScale())

    def getText(self):
        return self.text

    def getActive(self):
        return self.active

    def setFont(self, font):
        self.font = pg.font.Font(font, self.text_size)

    def setTextSize(self, text_size):
        self.text_size = text_size

    def setText(self, text):
        self.text = text

    def setRectY(self, y):
        self.rect = pg.Rect(self.x, y, self.w, self.h)

    def handle_event_send(self, event):
        self.txt_surface = self.font.render(self.text, True, self.color)
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_BACKSPACE:
                self.text = self.text[:-1]
            elif len(self.text) <= self.max_str_len and event.unicode.encode().isalpha() or event.unicode.isdigit() or event.unicode in "!@#$%^&*()-+?_=,<>/" or event.unicode == ' ':
                self.text += event.unicode

    def handle_event_save(self, event):
        self.resize()
        if event.type == pg.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            self.color = self.active_color if self.active else self.inactive_color

        if self.active:
            self.txt_surface = self.font.render(self.text, True, self.color)
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_RETURN:
                    self.color = self.inactive_color
                    self.active = False
                elif event.key == pg.K_BACKSPACE:
                    self.text = self.text[:-1]
                elif len(self.text) <= self.max_str_len and event.unicode.encode().isalpha() or event.unicode.isdigit() or event.unicode in "!@#$%^&*()-+?_=,<>/" or event.unicode == ' ':
                    self.text += event.unicode
        else:
            self.color = self.inactive_color
            self.txt_surface = self.font.render(self.text, True, self.color)

    def update(self, w):
        width = max(w, self.txt_surface.get_width() + 10)
        self.rect.w = width

    def draw(self, screen):
        screen.blit(self.txt_surface, (self.rect.x + 5, self.rect.y + 5))
        if self.border:
            pg.draw.rect(screen, self.color, self.rect, 4)

    def run(self, event, action):
        if action == 'SAVE':
            self.handle_event_save(event)
        elif action == 'SEND':
            self.handle_event_send(event)
        self.draw(self.screen.getScreen())
        self.update(self.w * self.screen.getWidthScale())

    def resize(self):
        self.font = pg.font.Font(self.font_raw, int(self.text_size * self.screen.getHeightScale()))
        self.txt_surface = self.font.render(self.text, True, self.color)
        self.rect = pg.Rect(self.x * self.screen.getWidthScale(), self.y * self.screen.getHeightScale(), self.w * self.screen.getWidthScale(), self.h * self.screen.getHeightScale())

class Pen(object):
    def __init__(self):
        self.COLOR = (0, 0, 0)
        self.pen_thickness = 10

    def getColor(self):
        return self.COLOR

    def getPenThickness(self):
        return self.pen_thickness

    def setColor(self, COLOR_RGB):
        self.COLOR = COLOR_RGB

    def setPenThickness(self, pen_thickness):
        self.pen_thickness = pen_thickness

    def flood_fill(self, screen, pos):
        arr = pg.surfarray.array3d(screen.getScreen())
        swapPoint = (pos[1], pos[0])
        cv2.floodFill(arr, None, swapPoint, self.COLOR)
        pg.surfarray.blit_array(screen.getScreen(), arr)

class Slider(object):
    def __init__(self, screen, x, y, w, h):
        self.screen = screen
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.volume = 50
        self.sliderRect = pg.Rect(self.x * self.screen.getWidthScale(), self.y * self.screen.getHeightScale(), self.w * self.screen.getWidthScale(), self.h * self.screen.getHeightScale())
        self.circle_x = int(self.x * self.screen.getWidthScale() + self.sliderRect.w / 2)

    def getVolume(self):
        return self.volume

    def setVolume(self, volume):
        self.volume = volume

    def draw(self, screen):
        pg.draw.rect(screen, (255, 255, 255), self.sliderRect)
        pg.draw.circle(screen, (255, 240, 255), (self.circle_x, (self.sliderRect.h / 2 + self.sliderRect.y)), self.sliderRect.h * 1.5)

    def update_volume(self, x):
        if x < self.sliderRect.x:
            self.volume = 0
        elif x > self.sliderRect.x + self.sliderRect.w:
            self.volume = 100
        else:
            self.volume = int((x - self.sliderRect.x) / float(self.sliderRect.w) * 100)

    def on_slider(self, x, y):
        if self.on_slider_hold(x, y) or self.sliderRect.x <= x <= self.sliderRect.x + self.sliderRect.w and self.sliderRect.y <= y <= self.sliderRect.y + self.sliderRect.h:
            return True
        else:
            return False

    def on_slider_hold(self, x, y):
        if ((x - self.circle_x) * (x - self.circle_x) + (y - (self.sliderRect.y + self.sliderRect.h / 2)) * (y - (self.sliderRect.y + self.sliderRect.h / 2))) <= (self.sliderRect.h * 1.5) * (self.sliderRect.h * 1.5):
            return True
        else:
            return False

    def handle_event(self, x, y):
        if self.on_slider_hold(x, y) and pg.mouse.get_pressed()[0] or self.on_slider(x, y) and pg.mouse.get_pressed()[0]:
            if x < self.sliderRect.x:
                self.circle_x = self.sliderRect.x
            elif x > self.sliderRect.x + self.sliderRect.w:
                self.circle_x = self.sliderRect.x + self.sliderRect.w
            else:
                self.circle_x = x
            self.update_volume(x)

        self.draw(self.screen.getScreen())

    def slider_update(self):
        self.sliderRect = pg.Rect(self.x * self.screen.getWidthScale(), self.y * self.screen.getHeightScale(), self.w * self.screen.getWidthScale(), self.h * self.screen.getHeightScale())
        self.circle_x = int(self.x * self.screen.getWidthScale() + self.sliderRect.w / 2)
        self.volume = 50

class Client(object):
    def __init__(self, client, addr=None):
        self.client = client
        self.nickname = ''
        if addr is not None:
            self.addr = addr
        self.index = None
        self.lobby = None
        self.was_painter = False
        self.client_id = str(uuid.uuid4())
        self.client_status = 'In_Lobby_Picker'
        self.score = 0
        self.emotion = 'neutral'


    def __repr__(self):
        return self.nickname

    def getNickname(self):
        return self.nickname

    def getAddr(self):
        return self.addr

    def getClient(self):
        return self.client

    def getLobby(self):
        return self.lobby

    def getIndex(self):
        return self.index

    def getClientID(self):
        return self.client_id

    def getWasPainter(self):
        return self.was_painter

    def getScore(self):
        return self.score

    def getClientStatus(self):
        return self.client_status
    
    def getEmotion(self):
        return self.emotion

    def setIndex(self, index):
        self.index = index

    def setWasPainter(self, was_painter):
        self.was_painter = was_painter

    def setLobby(self, lobby):
        self.lobby = lobby

    def setNickname(self, nickname):
        self.nickname = nickname

    def setScore(self, score):
        self.score = score

    def setClientStatus(self, client_status):
        self.client_status = client_status

    def setEmotion(self, emotion):
        self.emotion = emotion    

    def close(self):
        self.client.close()

class Lobby(object):
    def __init__(self, lobby_owner, lobby_name, lobby_password=None):
        self.lobby_owner = lobby_owner
        self.lobby_name = lobby_name
        self.lobby_password = lobby_password
        self.game_status = 'INACTIVE'
        self.players_list = [self.lobby_owner]
        self.words = self.getRandomWord(6)

    def getLobbyOwner(self):
        return self.lobby_owner

    def getPlayersList(self):
        return self.players_list

    def getLobbySpecs(self):
        return self.lobby_name, self.lobby_password

    def getGameStatus(self):
        return self.game_status

    def getWords(self):
        return self.words

    def setLobbyOwner(self, owner):
        self.lobby_owner = owner

    def setWords(self, words):
        self.words = words

    def setGameStatus(self, game_status):
        self.game_status = game_status

    def appendPlayersList(self, player):
        self.players_list.append(player)

    def removePlayersList(self, player):
        if player in self.players_list:
            self.players_list.remove(player)

    @staticmethod
    def getRandomWord(x=1):
        with open('assets\\words.txt', "r") as f:
            words = []
            for line in f:
                words.append(line[0:-1])
        return sample(words, x)