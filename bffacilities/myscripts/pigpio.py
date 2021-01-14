import RPi.GPIO as GPIO
import time
import random
import subprocess as sp
from datetime import datetime as dt
from bffacilities import createLogger
import os.path as osp
secs = random.randint(60, 200)
logger = createLogger("gpio", stream=True)

if osp.exists("/home/pi/rwfile.py"):
    p = sp.Popen(["python3", "/home/pi/rwfile.py"])

# GPIO.setmode(GPIO.BOARD)
GPIO.setmode(GPIO.BCM)
pin = 20
GPIO.setup(pin, GPIO.OUT)
logger.info(f"Secs: {secs}")
time.sleep(secs)
logger.info(f"Start gpio hight")
GPIO.output(pin, GPIO.HIGH)
time.sleep(30)
exit()
while True:
    GPIO.output(pin, GPIO.LOW)
    time.sleep(10)
    GPIO.output(pin, GPIO.HIGH)
    time.sleep(10)
