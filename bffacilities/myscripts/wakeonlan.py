__version__ = '0.0.4'
import socket
import struct
import argparse
from bffacilities._frame import Frame, common_entry
from bffacilities.utils import createLogger, initGetText

from pathlib import Path
import json
_BASE_DIR = Path.home() / ".shells" # Basic Path

tr = initGetText("wakeonlan")
logger = createLogger("bff", stream=True, savefile=False)

class LanWaker(Frame):
    DefaultLoc = _BASE_DIR / "wakeonlan.txt"
    def __init__(self):
        super().__init__()
    
    def readConfig(self, file):
        pass

    def initArgs(self):
        """ @return ArgumentParser """
        parser = argparse.ArgumentParser(prog="wakeOnLan", description=tr("Used for wake computer on lan."))
        parser.add_argument("-a", "--address", help=tr("specify broadcast ip (v4) address"), default="255.255.255.255")
        parser.add_argument("-p", "--port", type=int, help=tr("specify port"), default=7)
        parser.add_argument("-m", "--mac", type=str, help=tr("specify mac address"), default="FF:FF:FF:FF:FF:FF")
        parser.add_argument("-n", "--name", type=str, help=tr("specify the name of the machine, if specified, it will be recorded."))
        parser.add_argument("-l", "--list", action="store_true", help=tr("List recorded machine."))
        parser.add_argument("-r", "--repeat", type=str, help=tr("Use recorded machine's info."))
        parser.add_argument("-c", "--clear", action="store_true", help=tr("Clear recorded machine's info."))
        parser.add_argument("-V", "--version", help=tr("Print this script version"), 
            action="version", version=f'wakeOnLan {__version__}')

        self.parser = parser
    # def _read(self, )
    def parseArgs(self, argv):
        args = vars(self.parser.parse_args(argv))
        listIt = args["list"]
        repeat = args["repeat"]
        if listIt:
            try:
                with open(LanWaker.DefaultLoc, "r", encoding="utf-8") as f:
                    for l in f:
                        print(l)
            except:
                print("[List] Error while reading local records.")
            return
        if args["clear"]:
            try:
                with open(LanWaker.DefaultLoc, "w", encoding="utf-8") as f:
                    f.write("")
            except:
                pass
            return
        if repeat:
            if self.findName(repeat):
                self.wake(args["address"], args["port"], mac = args["mac"])
            return

        mac = args["mac"]
        mac = mac.replace(":", "")
        mac = mac.replace("-", "")
        mac = mac.upper()
        self.wake(args["address"], args["port"], mac)
        name = args["name"]
        if name is not None and len(name) > 0 and not self.findName(name):
            with open(LanWaker.DefaultLoc, "a", encoding="utf-8") as f:
                content = json.dumps({
                    "mac": mac,
                    "address": args["address"],
                    "port": args["port"],
                    "name": name
                })
                f.write(f"{content}\n")

    def findName(self, name):
        try:
            with open(LanWaker.DefaultLoc, "r", encoding="utf-8") as f:
                for l in f:
                    args = json.loads(l.strip())
                    if args['name'] == name:
                        return True
                        # break
        except Exception as e:
            print("[FN] Error while reading local records: ", e)
        
        return False
        
    def wake(self, addr, port, mac):
        mac_data = []
        try:
            for i in range(0, 12, 2):
                mac_data.append(int(mac[i:i+2], 16))
        except:
            logger.error(tr("Wrong Mac Address"))
            self.parser.print_help()
        packet = struct.pack("!BBBBBB", 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF)
        packet_mac = struct.pack("!BBBBBB", *mac_data)
        for i in range(0, 16):
            packet += packet_mac
        # print("len: ", len(packet), "data: ", packet)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            sock.sendto(packet, (addr, port))
            logger.info(f"{tr('Send wake packet to:')} {addr}:{port}, {mac}")
        finally:
            sock.close()
            
main = common_entry(LanWaker)