from classes   import Client, Lobby, Pen
from threading import Thread
from random    import sample
import socket
import pickle
import sys

def main():
    IP = '127.0.0.1'
    PORT = 12356
    server = Server(IP, PORT)
    server.listen()
    print(f"_____________________________________________\n[STARTING]  Server has been established...\n[LISTENING] Server is listening on {IP}:{PORT}\n_____________________________________________\n")
    server.server_management()
    server.close()

class Server(object):
    def __init__(self, IP, PORT):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addr = (IP, PORT)
        self.server.bind(self.addr)
        self.clients = []
        self.threads = []
        self.active_conns = 0
        self.lobby_list = []
        self.lobby_count = 0

    def server_management(self):
        for client in self.clients:
            client.close()
        del self.clients[:]
        
        self.server.settimeout(1.0)
        try:
            while True:
                try:
                    client, addr = self.server.accept()
                except socket.timeout:
                    continue

                cli = Client(client, addr)
                self.clients.append(cli)
                cli.setIndex(self.clients.index(cli))
                
                self.send(cli, ['NICK', cli.getClientID()], 2)
                nickname = self.recv(cli)
                self.clients[cli.getIndex()].setNickname(nickname)
                self.active_conns += 1

                print(f"[NEW CONNECTION] Address: [{str(addr[0])}:{str(addr[1])}] | Nickname: {self.clients[cli.getIndex()].getNickname()} | Active Connections: {self.active_conns}")

                t = Thread(target=self.handle_client, args=(cli,))
                t.daemon = True
                t.start()
                self.threads.append(t)
        except KeyboardInterrupt:
            for client in self.clients:
                try:
                    client.close()
                except:
                    pass
            self.server.close()
            sys.exit(0)

    def send(self, c1, data, to=0 or 1 or 2):
        """
        to = 0 -> send to every client in the lobby
        to = 1 -> send to every client in the lobby except c1
        to = 2 -> send to c1
        """
        try:
            try:
                for client in self.clients:
                    if c1 == client:
                        c1.setIndex(self.clients.index(client))
                for lobby in self.lobby_list:
                    if c1 in lobby.getPlayersList():
                        c1.setLobby(self.lobby_list.index(lobby) + 1)
            except Exception as e:
                print(e)
            if to == 0:
                for client in self.lobby_list[c1.getLobby() - 1].getPlayersList():
                    if client is not None:
                        client.getClient().sendall(pickle.dumps(data))
            elif to == 1:
                for client in self.lobby_list[c1.getLobby() - 1].getPlayersList():
                    if client != c1 and client is not None:
                        client.getClient().sendall(pickle.dumps(data))
            elif to == 2:
                c1.getClient().sendall(pickle.dumps(data))
        except Exception as e:
            print(f"[ERROR] An error occurred while trying to 'send':\n{e}")

    @staticmethod
    def recv(c1):
        try:
            try:
                return pickle.loads(c1.getClient().recv(1024 * 2))
            except:
                pass
        except Exception as e:
            print(f"[ERROR] An error occurred while trying to 'recieve':\n{e}")

    def server_cmd_dict(self):
        return {'ANNOUNCE_WINNER': self.cmd_announceWinner,
                'LOBBIES_SPECS': self.cmd_lobbySpecs,
                'CREATE_LOBBY' : self.cmd_createLobby,
                'ROUND_OVER': self.cmd_roundOver,
                'JOIN_LOBBY': self.cmd_joinLobby,
                'START_GAME': self.cmd_startGame,
                'COUNTDOWN' : self.cmd_countdown,
                'CHAT_MSG'  : self.cmd_chatMsg,
                'SCORE': self.cmd_score}

    def cmd_announceWinner(self, c1):
        if c1 == self.lobby_list[c1.getLobby()-1].getPlayersList()[0]:
            winner = None
            nick_list = self.getPlayersList(c1)
            del nick_list[0]
            for player in nick_list:
                if winner is None:
                    winner = player
                else:
                    if int(winner[1]) < int(player[1]):
                        winner = player
            self.send(c1, ['ANNOUNCE_WINNER', winner[0]], 0)
            self.lobby_list[c1.getLobby()-1].setWords(Lobby.getRandomWord(6))
            self.send(c1, ['WORDS', self.lobby_list[c1.getLobby()-1].getWords()], 0)
            self.lobby_list[c1.getLobby()-1].setGameStatus('INACTIVE')
            for client in self.clients and client.getClientStatus() == 'In_Lobby_Picker':
                self.cmd_lobbySpecs(client)

    def cmd_chatMsg(self, c1, msg):
        self.send(c1, msg, 0)

    def cmd_countdown(self, c1, msg):
        self.send(c1, msg, 1)

    def cmd_roundOver(self, c1, msg):
        if len(self.lobby_list[c1.getLobby() - 1].getPlayersList()) > 0:
            for client in self.lobby_list[c1.getLobby() - 1].getPlayersList():
                if c1 == client:
                    client.setWasPainter(True)
                    if msg[1] is not None:
                        client.setScore(str(msg[1]))
                    break
            painter = self.getRandomPainter(c1)
            self.send(c1, ['NEXT_ROUND', painter.getClientID(), self.getPlayersList(c1), str(painter.getNickname())], 0)
            words = self.lobby_list[c1.getLobby() - 1].getWords()
            if msg[1] is None:
                words.append(Lobby.getRandomWord(1)[0])
            del words[0]
            self.send(c1, ['WORDS', words], 0)

    def cmd_createLobby(self, c1, msg):
        self.lobby_list.append(Lobby(c1, msg[1], msg[2]))
        self.lobby_count += 1
        c1.setLobby(self.lobby_count)
        c1.setClientStatus('In_Game')
        print(f"[NEW LOBBY] A new lobby has been formed | Lobby {c1.getLobby()} | Owner: {c1.getNickname()}")
        self.send(c1, ['GAME_PREP', self.getPlayersList(c1), self.lobby_list[c1.getLobby() - 1].getWords(), self.lobby_list[c1.getLobby()-1].getLobbyOwner().getClientID()], 0)
        for client in self.clients:
            if client.getClientStatus() == 'In_Lobby_Picker':
                self.cmd_lobbySpecs(client)

    def cmd_joinLobby(self, c1, msg):
        for lobby in self.lobby_list:
            if lobby.getGameStatus() == 'INACTIVE' and len(lobby.getPlayersList()) < 6:
                if msg[1] == lobby.getLobbySpecs()[0] and msg[2] == lobby.getLobbySpecs()[1]:
                    c1.setLobby(self.lobby_list.index(lobby)+1)
                    c1.setClientStatus('In_Game')
                    self.lobby_list[c1.getLobby() - 1].appendPlayersList(c1)
                    self.send(c1, ['GAME_PREP', self.getPlayersList(c1), self.lobby_list[c1.getLobby() - 1].getWords(), self.lobby_list[c1.getLobby()-1].getLobbyOwner().getClientID()], 0)
        for client in self.clients:
            if client.getClientStatus() == 'In_Lobby_Picker':
                self.cmd_lobbySpecs(client)

    def cmd_lobbySpecs(self, c1):
        self.send(c1, self.getLobbyListSpecs(), 2)

    def cmd_startGame(self, c1):
        painter = self.getRandomPainter(c1)
        self.send(c1, ['START_GAME', painter.getClientID(), str(painter.getNickname())], 0)
        self.lobby_list[c1.getLobby()-1].setGameStatus('ACTIVE')
        for client in self.clients:
            if client.getClientStatus() == 'In_Lobby_Picker':
                self.cmd_lobbySpecs(client)

    @staticmethod
    def cmd_score(c1, msg):
        c1.setScore(str(msg[1]))

    def handle_client(self, c1):
        while True:
            for client in self.clients:
                if c1 == client:
                    c1.setIndex(self.clients.index(client))
            for lobby in self.lobby_list:
                if c1 in lobby.getPlayersList():
                    c1.setLobby(self.lobby_list.index(lobby) + 1)
            msg = self.recv(c1)
            if type(msg) is Pen:
                pass
            else:
                if type(msg) is list:
                    info = msg[0]
                    if info == "UPDATE_EMOTION":
                        c1.setEmotion(msg[1])
                        self.broadcast_emotion_updates(c1)
                    elif info in self.server_cmd_dict().keys():
                        self.server_cmd_dict()[info](c1, msg)
                elif type(msg) is str:
                    if msg in self.server_cmd_dict().keys():
                        self.server_cmd_dict()[msg](c1)
                    elif 'DISCONNECT' == msg:
                        break
        self.quit_progress(c1)

    def getPlayersList(self, c1):
        nick_list = ['NICKNAME_LIST']
        if self.lobby_count > 0:
            for client in self.lobby_list[c1.getLobby() - 1].getPlayersList():
                nick_list.append((client.getNickname(), client.getScore()))
        return nick_list

    def getLobbyListSpecs(self):
        lobby_list_specs = ['LOBBIES_SPECS']
        for lobby in self.lobby_list:
            if lobby.getLobbySpecs()[1] is None:
                locked = False
            else:
                locked = True
            lobby_list_specs.append([lobby.getLobbySpecs()[0], str(lobby.getLobbyOwner()),
                                     len(lobby.getPlayersList()), locked, lobby.getGameStatus()])
        return lobby_list_specs

    def getRandomPainter(self, c1):
        approved_clients = []
        self.append_approved_clients(c1, approved_clients)
        if len(approved_clients) == 0:
            self.reset_client_approval(c1)
            self.append_approved_clients(c1, approved_clients)
        return sample(approved_clients, 1)[0]

    def reset_client_approval(self, c1):
        for client in self.lobby_list[c1.getLobby() - 1].getPlayersList():
            if c1 != client:
                client.setWasPainter(False)

    def append_approved_clients(self, c1, approved_clients):
        for client in self.lobby_list[c1.getLobby() - 1].getPlayersList():
            if not client.getWasPainter():
                approved_clients.append(client)

    def quit_progress(self, c1):
        if self.active_conns > 0:
            self.active_conns -= 1
        for lobby in self.lobby_list:
            if c1 in lobby.getPlayersList():
                c1.setLobby(self.lobby_list.index(lobby) + 1)
        for client in self.clients:
            if c1 == client:
                c1.setIndex(self.clients.index(client))
        print(f"[DISCONNECTION]  Address: [{c1.getAddr()[0]}:{str(c1.getAddr()[1])}] | Nickname: {c1.getNickname()} | Lobby: {c1.getLobby()} | Active Connections: {self.active_conns}")
        if self.lobby_count > 0 and c1.getLobby() is not None:
            self.lobby_list[c1.getLobby() - 1].removePlayersList(c1)
            if len(self.lobby_list[c1.getLobby()-1].getPlayersList()) > 0:
                self.send(c1, self.getPlayersList(c1), 1)
                if c1 == self.lobby_list[c1.getLobby()-1].getLobbyOwner():
                    self.lobby_list[c1.getLobby() - 1].setLobbyOwner(self.lobby_list[c1.getLobby()-1].getPlayersList()[0])
                    self.send(c1, ['SET_OWNER_ID', self.lobby_list[c1.getLobby() - 1].getLobbyOwner().getClientID()], 1)
            if len(self.lobby_list[c1.getLobby() - 1].getPlayersList()) == 0:
                del self.lobby_list[c1.getLobby() - 1]
                self.lobby_count -= 1
                print(f"[LOBBY REMOVAL] Lobby has been deconstructed | Lobby {c1.getLobby()} | Remaining Lobbies: {self.lobby_count}")
        del self.clients[c1.getIndex()]
        del self.threads[c1.getIndex()]
        c1.close()
        for client in self.clients:
            if client.getClientStatus() == 'In_Lobby_Picker':
                self.cmd_lobbySpecs(client)

    def listen(self):
        self.server.listen()

    def close(self):
        self.server.close()

if __name__ == '__main__':
    main()
