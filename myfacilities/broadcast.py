#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
## 如何查找特定设备的 IP
有几种方法在局域网中找到某个设备（设为设备 A）的 IP 地址：
1. 在设备 A 上运行一段程序，该程序每隔一段时间向局域网中发送广播包（UDP 广播包），（设备 B）上运行另一个程序监听相应的端口，当接收到特定格式的消息时认为收到正确的消息，此时在命令行中打印出来的的远程设备的 IP 地址即为需要的 IP。
2. 在设备 A 上运行一段程序，该程序监听预先约定好的端口，在设备 B 上向所有 IP 地址的该端口（广播）发送消息，远程设备回复时即可得到对应的 IP。
注意发送 UDP 的广播包时，将 IP 地址设为 "255.255.255.255" 即可广播到整个网络，设为 "192.168.0.255" 可广播到 "192.168.0.0/24" 的网络。如果设为其他的 IP 地址如 “192.168.255.255” 则程序运行时会报错。
## 代码地址：
 
https://gist.github.com/BriFuture/5789fef5db9d233d2a405c0cfd6a8462

update: 19-03-01 
1. autematically check whether SERVER_RECV_PORT is useable, and if it is not useable and this script is running under server mode,
then it won't start UdpServer, 
2. replace print function with logger, when running under local mode, logger will stream message to stdout
"""
# 功能：发送广播包或接受心跳包

__version__ = "0.0.5"

SERVER_RECV_PORT = 7000
LOCAL_RECV_PORT = 7007
import socket

from . import createLogger

logger = None


def local_machine():
    """send broadcast udp packet to see who will response
    """
    con = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    con.settimeout(3)
    con.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    con.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    
    serIp = "255.255.255.255"
    con.sendto(b"'test'", (serIp, SERVER_RECV_PORT))
    logger.info("Wait Response")
    try:
        msg, addr = con.recvfrom(1024)
        logger.info(f"Recv from {addr[0]}: {msg}")
    except Exception as e:
        logger.warning(e)
    con.close()


from socketserver import UDPServer, BaseRequestHandler
import time
import json

MsgHeaderLen = len('"camera_1":')
class UdpHandler(BaseRequestHandler):
    
    def handle(self):
        msg, sock = self.request
        resp = time.ctime()
        # msgObject = json.loads(msg[MsgHeaderLen:])
        logger.info(f"recved from {self.client_address}: {msg}")
        # sock.sendto(resp.encode("ascii"), (self.client_address[0], LOCAL_RECV_PORT))
        sock.sendto(resp.encode("ascii"), self.client_address)


class Interval(object):
    def __init__(self):
        import socket
        con = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        con.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        con.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        con.settimeout(5)
        self.con = con
    
    def send(self):
        try:
            self.con.sendto(b"HeartBeat", ("255.255.255.255", LOCAL_RECV_PORT))
            msg, addr = self.con.recvfrom(1024)
            logger.info(f"Recv from {addr[0]}: {msg}")
        except Exception as e:
            logger.warning("Heart beat sent but no response: {}".format(e))

    def serve_forever(self):
        while True:
            self.send()
            time.sleep(10)


def run_server():
    from threading import Thread
    inter = Interval()
    t = Thread(target=inter.serve_forever, daemon=True)
    t.start()
    server = UDPServer(("", SERVER_RECV_PORT), UdpHandler)
    try:
        server.serve_forever() 
    except Exception as e:
        logger.warning(e)

def isServerPortAvailable(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(3)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 0)
    sock.sendto(b"'test port available'", ("127.0.0.1", SERVER_RECV_PORT))
    try:
        msg, addr = sock.recvfrom(1024)
        logger.info("{} is not available".format(port))
    except Exception as e:
        logger.info("{} is available".format(port))
        logger.warning(e)
        sock.close()
        return True
    
    sock.close()
    return False

def main():
    import sys
    isLocal = False
    if len(sys.argv) > 1 and sys.argv[1] == "local":
        isLocal = True

    global logger
    if isLocal:
        logger = createLogger("broadcast.log", stream=True)
        logger.info("Running Local Mode")
        local_machine()
        server = UDPServer(("", LOCAL_RECV_PORT), UdpHandler)
        try:
            server.serve_forever() 
        except Exception as e:
            logger.warning(e)
    else:
        logger = createLogger("broadcast.log", stream=False)
        if isServerPortAvailable(SERVER_RECV_PORT):
            logger.info("Running Server Mode")
            run_server()

if __name__ == '__main__':
    main()