
import socket

from crypto import KeyManager, DES

class Server:
    def __init__(self, addr, port, buffer_size=1024):
        self.addr = addr
        self.port = port
        self.buffer_size = buffer_size

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((self.addr, self.port))
        self.s.listen(1)
        self.conn, self.addr = self.s.accept()

    def send(self, msg_bytes: bytes):
        self.conn.send(msg_bytes)

    def recv(self, buffer_size=None) -> bytes:
        if buffer_size is None:
            buffer_size = self.buffer_size
        msg_bytes = self.conn.recv(buffer_size)

        return msg_bytes

    def close(self):
        self.conn.close()


if __name__ == '__main__':
    server = Server('localhost', 9999)
    key = KeyManager.read_key('key.txt')
    subkeys = DES.key_generation(key)
    des = DES(key)

    print(f"Connected by {server.addr}")

    DES.key_generation(key)
    
    while True:
        
        data = server.recv(1024).decode()
        print(des.decrypt(data))

        msg = input('> ')
        if msg == 'exit':
            break

        server.send(des.encrypt(msg))

    server.close()
