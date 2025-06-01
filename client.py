from assets_loader import *
from hand_detect   import HandDetector
from classes import Slider
from game    import Game

import sys

class Menu:
    def __init__(self):
        self.screen = screen
        self.selected_resolution = f'{self.screen.getWidth()} x {self.screen.getHeight()}'
        self.nickname = ''
        self.running = False
        self.x, self.y = 0, 0
        self.page = 'Main'
        self.music_slider = Slider(self.screen, 765, 655, 391, 15)
        self.music = Music(0.01)
        self.game = Game(self.screen, self.music)

    def run(self):
        while True:
            try:
                while not self.running:
                    for event in pg.event.get():
                        self.x, self.y = pg.mouse.get_pos()
                        if self.page == 'Main' and self.running == False:
                            self.main_page(event)
                        elif self.page == 'About' and self.running == False:
                            self.about_page(event)
                        elif self.page == 'Settings' and self.running == False:
                            self.settings_page(event)
                        elif self.page == 'Game':
                            self.running = True

                        self.music.display_icon(self.screen, mute_on, mute_off)
                        if event.type == pg.QUIT:
                            pg.quit()
                            sys.exit()
                        self.screen.flip()
                else:
                    try:
                        self.game.game_manager()
                        self.nickname = nickname_box.getText()
                        self.game = Game(self.screen, self.music, self.nickname)
                        self.running = False
                        nickname_box.resize()
                    except Exception as e:
                        self.running = False
                        print(e)
            except Exception as e:
                print(e)
                pass

    def start_button(self, event):
        hover_sound.play_sound_static()
        self.screen.blit(start_clicked, (0, 0))
        if event.type == pg.MOUSEBUTTONDOWN:
            self.nickname = nickname_box.getText()
            if len(self.nickname) > 0:
                click_sound.play_sound_static()
                if ' ' in self.nickname:
                    self.nickname = self.nickname.replace(' ', '_')
                self.game.setPlayer(self.nickname)
                self.running = True
            else:
                error_sound.play_sound_static()
                error_sound.reset()

    def about_button(self, event):
        hover_sound.play_sound_static()
        self.screen.blit(about_clicked, (0, 0))
        if event.type == pg.MOUSEBUTTONDOWN:
            click_sound.play_sound(event)
            self.page = 'About'

    def exit_button(self, event):
        hover_sound.play_sound_static()
        self.screen.blit(exit_clicked, (0, 0))
        if event.type == pg.MOUSEBUTTONDOWN:
            click_sound.play_sound(event)
            pg.quit()
            sys.exit()

    def music_button(self, event):
        hover_sound.play_sound_static()
        if self.music.getPlay():
            self.screen.blit(mute_on_clicked, (0, 0))
        else:
            self.screen.blit(mute_off_clicked, (0, 0))
        click_sound.play_sound(event)
        self.music.play_music(event)

    def settings_button(self, event):
        hover_sound.play_sound_static()
        self.screen.blit(settings_clicked, (0, 0))
        if event.type == pg.MOUSEBUTTONDOWN:
            click_sound.play_sound(event)
            self.page = 'Settings'

    def back_button(self, event):
        hover_sound.play_sound_static()
        self.screen.blit(back_icon, (0, 0))
        if event.type == pg.MOUSEBUTTONDOWN:
            click_sound.play_sound(event)
            self.selected_resolution = f'{self.screen.getWidth()} x {self.screen.getHeight()}'
            self.page = 'Main'

    def left_arrow_button(self, event):
        hover_sound.play_sound_static()
        self.screen.blit(left_settings_arrow, (0, 0))
        if event.type == pg.MOUSEBUTTONDOWN and self.screen.getResolutionOptions().index(self.selected_resolution) - 1 >= 0:
            self.selected_resolution = self.screen.getResolutionOptions()[self.screen.getResolutionOptions().index(self.selected_resolution) - 1]

    def right_arrow_button(self, event):
        hover_sound.play_sound_static()
        self.screen.blit(right_settings_arrow, (0, 0))
        if event.type == pg.MOUSEBUTTONDOWN and self.screen.getResolutionOptions().index(self.selected_resolution) + 1 < len(self.screen.getResolutionOptions()):
            self.selected_resolution = self.screen.getResolutionOptions()[self.screen.getResolutionOptions().index(self.selected_resolution) + 1]

    def apply_button(self, event):
        hover_sound.play_sound_static()
        self.screen.blit(apply_clicked, (0, 0))
        if event.type == pg.MOUSEBUTTONDOWN and f'{self.screen.getWidth()} x {self.screen.getHeight()}' != self.selected_resolution:
            res = self.selected_resolution.split(' x ')
            self.screen_resize(int(res[0]), int(res[1]))

    def settings_page(self, event):
        self.screen.blit(settings_background, (0, 0))
        self.music_slider.handle_event(self.x, self.y)
        self.music.setVolume(self.music_slider.getVolume() / 1000)
        self.screen.draw_text(str(self.selected_resolution), (255, 255, 255), 'assets\\fonts\\ChildrenSans.ttf', 100, 838, 445)

        if 29 * self.screen.getWidthScale() < self.x < 147 * self.screen.getWidthScale() and 31 * self.screen.getHeightScale() < self.y < 117 * self.screen.getHeightScale():
            self.music_button(event)
        elif 726 * self.screen.getWidthScale() < self.x < 793 * self.screen.getWidthScale() and 429 * self.screen.getHeightScale() < self.y < 502 * self.screen.getHeightScale():
            self.left_arrow_button(event)
        elif 1128 * self.screen.getWidthScale() < self.x < 1195 * self.screen.getWidthScale() and 429 * self.screen.getHeightScale() < self.y < 502 * self.screen.getHeightScale():
            self.right_arrow_button(event)
        elif 1652 * self.screen.getWidthScale() < self.x < 1875 * self.screen.getWidthScale() and 954 * self.screen.getHeightScale() < self.y < 1026 * self.screen.getHeightScale():
            self.apply_button(event)
        elif 150 * self.screen.getWidthScale() < self.x < 207 * self.screen.getWidthScale() and 32 * self.screen.getHeightScale() < self.y < 112 * self.screen.getHeightScale() or \
             207 * self.screen.getWidthScale() < self.x < 270 * self.screen.getWidthScale() and 58 * self.screen.getHeightScale() < self.y < 89  * self.screen.getHeightScale() or \
             251 * self.screen.getWidthScale() < self.x < 276 * self.screen.getWidthScale() and 89 * self.screen.getHeightScale() < self.y < 113 * self.screen.getHeightScale():
            self.back_button(event)
        else:
            self.reset()

    def main_page(self, event):
        self.screen.blit(main_menu_background, (0, 0))
        nickname_box.run(event, 'SAVE')
        if   775  * self.screen.getWidthScale() < self.x < 1140 * self.screen.getWidthScale() and 320 * self.screen.getHeightScale() < self.y < 430  * self.screen.getHeightScale():
            self.start_button(event)
        elif 826  * self.screen.getWidthScale() < self.x < 1090 * self.screen.getWidthScale() and 454 * self.screen.getHeightScale() < self.y < 532  * self.screen.getHeightScale():
            self.about_button(event)
        elif 1732 * self.screen.getWidthScale() < self.x < 1883 * self.screen.getWidthScale() and 892 * self.screen.getHeightScale() < self.y < 1043 * self.screen.getHeightScale():
            self.settings_button(event)
        elif 870  * self.screen.getWidthScale() < self.x < 1046 * self.screen.getWidthScale() and 540 * self.screen.getHeightScale() < self.y < 610  * self.screen.getHeightScale():
            self.exit_button(event)
        elif 29   * self.screen.getWidthScale() < self.x < 147  * self.screen.getWidthScale() and 31  * self.screen.getHeightScale() < self.y < 117  * self.screen.getHeightScale():
            self.music_button(event)
        else:
            self.reset()

    def about_page(self, event):
        self.screen.blit(about_background, (0, 0))
        if 29 * self.screen.getWidthScale() < self.x < 147 * self.screen.getWidthScale() and 31 * self.screen.getHeightScale() < self.y < 117 * self.screen.getHeightScale():
            self.music_button(event)
        elif 150 * self.screen.getWidthScale() < self.x < 207 * self.screen.getWidthScale() and 32 * self.screen.getHeightScale() < self.y < 112 * self.screen.getHeightScale() or \
             207 * self.screen.getWidthScale() < self.x < 270 * self.screen.getWidthScale() and 58 * self.screen.getHeightScale() < self.y <  89 * self.screen.getHeightScale() or \
             251 * self.screen.getWidthScale() < self.x < 276 * self.screen.getWidthScale() and 89 * self.screen.getHeightScale() < self.y < 113 * self.screen.getHeightScale():
            self.back_button(event)
        else:
            self.reset()

    def reset(self):
        if nickname_box.getActive():
            self.screen.blit(nickname_clicked, (0, 0))
        hover_sound.reset()
        click_sound.reset()

    def screen_resize(self, w, h):
        self.screen.screen_resize(w, h)
        nickname_box.resize()
        chat_input_box.resize()
        self.music_slider.slider_update()

if __name__ == '__main__':
    run = Menu()
    run.run()