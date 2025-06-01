from classes import *

WIDTH  = 960
HEIGHT = 540
screen = PgSetup(WIDTH, HEIGHT)

lobby_picker_background = screen.loadify('assets\\images\\interface\\lobby_handler\\lobby_handle_background.png')
main_menu_background = screen.loadify('assets\\images\\interface\\main_menu\\main_menu_background.png')
settings_background  = screen.loadify('assets\\images\\interface\\main_menu\\settings_background.png')
about_background     = screen.loadify('assets\\images\\interface\\main_menu\\about_background.png')

waiting_for_owner_to_start_interface = screen.loadify('assets\\images\\interface\\in_game\\waiting_for_the_owner_to_start_interface.png')
waiting_for_players_interface = screen.loadify('assets\\images\\interface\\in_game\\waiting_for_players_interface.png')
start_game_interface = screen.loadify('assets\\images\\interface\\in_game\\start_game_interface.png')
painter_interface    = screen.loadify('assets\\images\\interface\\in_game\\painter_game_interface.png')
guesser_interface    = screen.loadify('assets\\images\\interface\\in_game\\guesser_game_interface.png')
empty_interface = screen.loadify('assets\\images\\interface\\in_game\\empty_game_interface.png')

owner_start_game_clicked_icon = screen.loadify('assets\\images\\icons\\additional_icons\\start_game_icon.png')
timer_icon = screen.loadify('assets\\images\\icons\\additional_icons\\countdown_timer_icon.png')
crown_icon = screen.loadify('assets\\images\\icons\\additional_icons\\leading_crown_icon.png')
back_icon  = screen.loadify('assets\\images\\icons\\additional_icons\\back_to_menu_icon.png')

mute_off_clicked = screen.loadify('assets\\images\\icons\\mute\\mute_off_clicked.png')
mute_on_clicked  = screen.loadify('assets\\images\\icons\\mute\\mute_on_clicked.png')
mute_off = screen.loadify('assets\\images\\icons\\mute\\mute_off.png')
mute_on  = screen.loadify('assets\\images\\icons\\mute\\mute_on.png')

right_settings_arrow = screen.loadify('assets\\images\\icons\\additional_icons\\right_settings_arrow.png')
left_settings_arrow  = screen.loadify('assets\\images\\icons\\additional_icons\\left_settings_arrow.png')
settings_clicked = screen.loadify('assets\\images\\interface\\main_menu\\setting_clicked.png')
nickname_clicked = screen.loadify('assets\\images\\interface\\main_menu\\nickname_clicked.png')
start_clicked = screen.loadify('assets\\images\\interface\\main_menu\\start_clicked.png')
about_clicked = screen.loadify('assets\\images\\interface\\main_menu\\about_clicked.png')
apply_clicked = screen.loadify('assets\\images\\interface\\main_menu\\apply_clicked.png')
exit_clicked  = screen.loadify('assets\\images\\interface\\main_menu\\exit_clicked.png')

clear_button_clicked = screen.loadify('assets\\images\\icons\\actions\\clear_clicked_icon.png')
erase_button_clicked = screen.loadify('assets\\images\\icons\\actions\\erase_clicked_icon.png')
down_button_clicked  = screen.loadify('assets\\images\\icons\\actions\\down_clicked_icon.png')
fill_button_clicked  = screen.loadify('assets\\images\\icons\\actions\\fill_clicked_icon.png')
pen_button_clicked   = screen.loadify('assets\\images\\icons\\actions\\pen_clicked_icon.png')
up_button_clicked    = screen.loadify('assets\\images\\icons\\actions\\up_clicked_icon.png')

hide_locked_lobbies_checkmark = screen.loadify('assets\\images\\interface\\lobby_handler\\hide_locked_lobbies_checkmark.png')
hide_full_lobbies_checkmark   = screen.loadify('assets\\images\\interface\\lobby_handler\\hide_full_lobbies_checkmark.png')
lobby_picker_back_icon = screen.loadify('assets\\images\\interface\\lobby_handler\\back_clicked_icon.png')
create_icon_clicked = screen.loadify('assets\\images\\interface\\lobby_handler\\create_clicked_icon.png')
password_checkmark  = screen.loadify('assets\\images\\interface\\lobby_handler\\password_checkmark.png')
join_icon_clicked   = screen.loadify('assets\\images\\interface\\lobby_handler\\join_clicked_icon.png')
password_cover = screen.loadify('assets\\images\\interface\\lobby_handler\\password_cover.png')
lock_icon = screen.loadify('assets\\images\\interface\\lobby_handler\\lock_icon.png')

lobby_password_box = InputBox(screen, 845, 952, 360, 31, 'assets\\fonts\\ChildrenSans.ttf',  25, (0, 0, 0), (0, 0, 0), 30, False)
lobby_name_box = InputBox(screen, 845,  915, 360, 31, 'assets\\fonts\\ChildrenSans.ttf',     25, (0, 0, 0), (0, 0, 0), 30, False)
chat_input_box = InputBox(screen, 1443, 941, 462, 60, 'assets\\fonts\\Dosis-ExtraBold.ttf',  30, (3, 53, 93), (3, 53, 93), 21)
nickname_box   = InputBox(screen, 15,   975, 311, 60, 'assets\\fonts\\JungleAdventurer.ttf', 50, (255, 255, 255), (134, 134, 134), 8)

winner_sound = Sound(0.02, 'assets\\sounds\\winner_sound.wav')
error_sound  = Sound(0.02, 'assets\\sounds\\button_error_sound.wav')
click_sound  = Sound(0.2,  'assets\\sounds\\button_push_sound.wav')
hover_sound  = Sound(0.2,  'assets\\sounds\\button_hover_sound.wav')
pen_sound    = Sound(0.05, 'assets\\sounds\\pen_sound.wav')
