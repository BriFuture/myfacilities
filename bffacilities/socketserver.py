import json
import socket
import select
from .utils import createLogger
logger = createLogger('sockserver', savefile=False, stream=True)

class EventHandler:
    def fileno(self):
        'Return the associated file descriptor'
        raise ValueError('Not Implemented must implement')

    def handle_receive(self):
        'Perform the receive operation'
        pass

    def handle_send(self):
        'Send outgoing data'
        pass

    def handle_except(self):
        pass

class SocketClient(EventHandler):
    def __init__(self, sock, address, server, *args, **kwargs):
        self.sock = sock
        self.address = address
        self._connected = True
        self.sock.setblocking(False)
        self.server = server

    def fileno(self):
        return self.sock.fileno()
        
    def remove(self):
        self.sock.close()
        self.server.removeClient(self)

    def handle_receive(self):
        """Handle receive data, default processor
        subclass could replace this methods to receive binary data
        """
        try:
            msg = self.sock.recv(1024)
            if not msg:
                self.server.removeClient(self)
                return
            data = json.loads(msg.decode())
            self._parse(data)
        except Exception as e:
            logger.warning(f"[E] {e}")

    def handle_send(self):
        logger.warning("[D] not implemented")
        return super().handle_send()
    def handle_except(self):
        self.sock.close()
        self._connected = False
    def _parse(self, data):
        """子类实现解析即可
        """
        pass

class TcpSocketServer(EventHandler):
    ClientClass = SocketClient
    def __init__(self, ip, port, maxconn = 100):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((ip, port))
        logger.info(f"[TSS] Server Listening: {ip}:{port}")

        self.server.listen(maxconn)
        self.server.setblocking(False)
        self.clients = {}
        self.recvList = [ self ]
        self.sendList = []
        self.exceptList = [ self ]
        self.running = False

    def fileno(self):
        return self.server.fileno()

    def needs_receive(self):
        return True

    def serve_forever(self):
        self.running = True
        while self.running:
            can_recv, can_send, exc = select.select(self.recvList, self.sendList, self.exceptList, 0.5)
            for h in can_recv:
                h.handle_receive()
            for h in can_send:
                h.handle_send()
            for h in exc:
                logger.warning(f"Handleing Except when: {h}")
                h.handle_except()
                self.removeClient(h)

    def removeAll(self):
        for _, c in self.clients.items():
            c.sock.close()
            self.removeClient(c)

    def removeClient(self, client):
        if client == self: return

        logger.info(f"[Remove] {client.address}")
        try:
            self.recvList.remove(client)
            self.exceptList.remove(client)
            # self.sendList.remove(client)
        except: pass
        
        self.clients.pop(client.address, None)
        self._clientRemoved(client)

    def addClient(self, sock, address):
        c = TcpSocketServer.ClientClass(sock, address, self)
        self.clients[address] = (c)
        self.recvList.append(c)
        # self.sendList.append(c)
        self.exceptList.append(c)
        self._clientAdded(c)

    def _clientAdded(self, client):
        pass
    def _clientRemoved(self, client):
        pass

    def handle_receive(self):
        (sock, address) = self.server.accept()
        if address not in self.clients:
            self.addClient(sock, address)
        else:
            self.clients[address].sock = sock
            self.clients[address].address = address
            self.clients[address]._connected = True
        logger.info(f'[TSS] Accept From {address}, {len(self.clients)}')

class UdpSocketServer(EventHandler):
    """process: a callback function
    """
    def __init__(self, port, ip="0.0.0.0", maxconn=1000, process=None, **kwargs):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server.bind((ip, port))
        logger.info(f"[USS] Server Listening: {ip}:{port}")
        self.server.setblocking(False)
        self.clients = {}
        self.running = False
        self.bufferSize = kwargs.get("bufferSize", 1024)
        self.recvList = [ self ]
        self.sendList = []
        self.exceptList = [ self ]
        self.process = process

    def fileno(self):
        return self.server.fileno()

    def handle_receive(self):
        bytesAdressPair = self.server.recvfrom(self.bufferSize)

        message = bytesAdressPair[0]
        address = bytesAdressPair[1]
        if self.process is not None:
            self.process(message.decode(), address)
        else:
            print(f"{address}: {message}")

    def handle_send(self):
        pass

    def handle_except(self):
        pass
    def serve_forever(self):
        self.running = True
        while self.running:
            can_recv, can_send, exc = select.select(self.recvList, self.sendList, self.exceptList, 0.5)
            for h in can_recv:
                h.handle_receive()
            for h in can_send:
                h.handle_send()
            for h in exc:
                logger.warning(f"Handleing Except when: {h}")
                h.handle_except()
                # self.removeClient(h)
            
