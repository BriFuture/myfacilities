#!/bin/sh

net=`ip route | grep 'wlx0857009249d7' | wc -l`
log=/var/log/checknet.log
date > $log

resetNet() {
        wifiname=$1
        gateway='192.168.0.1'
        iwconfig $wifiname essid "Tenda_DaPing_2.4G"
        iwconfig $wifiname key 59260479
        iwconfig $wifiname mode Managed
        dhclient $wifiname &

        metrics=`ip route | grep 'proto dhcp' | awk '{print $9}'`
        for m in $metrics; do
                ip route del default via ${gateway} dev $wifiname proto dhcp metric $m
        done
        ip route add default via ${gateway} dev $wifiname proto dhcp metric 60
}
resetNet2() {
        wifi=$1
        ifconfig $wifi down
        ifconfig $wifi up
        echo "setting $wifi" >> $log
        killall wpa_supplicant >> $log
        killall dhclient >> $log
        echo "setting $wifi" >> $log
        wpa_supplicant  -i $wifi -c /etc/wpa_supplicant/wpa_sup.config -B &
        dhclient $wifi &
}

checkDns() {
        dns=`cat /etc/resolve.conf | grep '119.29.29.29' | wc -l`
        if [ $dns -eq 0 ]; then
                rm -rf /etc/resolve.conf
                echo 'nameserver 119.29.29.29' >> /etc/resolve.conf
        else
                echo 'dns is right' | tee >> $log
        fi

}
if [ $net -eq 0 ]; then
        echo 'not connected to internet' >> $log
        # ifconfig p2p0 down
        resetNet2 wlx0857009249d7 # wlan0
        echo 'Set Wifi internet' >> $log
        iwconfig wlx0857009249d7 >> $log
else
        echo 'Connected to internet' | tee >> $log
fi

checkDns
