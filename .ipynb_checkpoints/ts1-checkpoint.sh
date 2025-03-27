#!/bin/bash 

FILENAME="flist.txt"

LINES=$(cat $FILENAME)

for LINE in $LINES
do
    mv "test_fr/${LINE}" test_to/;
done

exit 