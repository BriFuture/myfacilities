#!/bin/sh
# Desktop
yum -y install  epel-release
yum -y groups install "Xfce"

startxfce4 &

# X11
yum -y install x11vnc
cp ../systemd/x11vnc.service /lib/systemd/system/x11vnc.service
cp ../config/X11-xorg.conf /etc/X11/xorg.conf