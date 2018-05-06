#!/bin/bash

for file in $1/*
do
    base=$(basename -- "$file")
    firstletter=${base:0:1}
    if [ $firstletter == "0" ]
    then
        mv $file $1/A_$base
        #echo "mv $file $1/A$rest"
    elif [ $firstletter == "1" ]
    then
        mv $file $1/A_$base
        #echo "mv $file $1/A$rest"
    elif [ $firstletter == "3" ]
    then
        mv $file $1/A_$base
        #echo "mv $file $1/A$rest"
    elif [ $firstletter == "2" ]
    then
        mv $file $1/B_$base
        #echo "mv $file $1/B$rest"
    elif [ $firstletter == "4" ]
    then
        mv $file $1/B_$base
        #echo "mv $file $1/B$rest"
    fi
done
