#!/bin/bash

port=$1
if [ -z $2  ]; then
        action='add-port'
else
        action=$2
        echo "$action"
fi
firewall-cmd --permanent --zone=public --$action=$port/tcp
firewall-cmd --reload
