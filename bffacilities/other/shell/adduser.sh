#!/bin/sh

username=$1

if [ -z "$username" ] ; then
        echo "Empty username"
        exit 1
fi


rand() {
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))
}


rnd=$(rand 10 99)

username="$username$rnd"
echo Username: $username


# mkdir "/home/$username"
adduser $username

cp /shell/.bashrc "/home/$username/"
cp /shell/.profile "/home/$username/"
# passwd $username
