#!/bin/bash

count=0

while [ $count -lt 100 ]
do
    dir="epoch_$count.smi"

    if [ ! -e $dir ]
    then
        let "count--"
        echo "$count cycle"
        break
    fi
    sed -i "/^ [0-9]\+$/d" $dir
    sed -i "/Xe/d" $dir

    let "count++"
done
