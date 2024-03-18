#!/bin/bash  

while getopts ":hr:l:" opt; do
    case $opt in
        r ) echo "Run Numbers - argument = $OPTARG "
            set -f # disable glob
            IFS=' ' # split on space characters
            array=($OPTARG) ;; # use the split+glob operator
        l ) echo "Latency range - argument = $OPTARG" ;;
        h ) helptext
            graceful_exit ;;
        * ) usage
            clean_up
            exit 1
    esac
done

echo "Number of arguments: ${#array[@]}"
echo -n "Arguments are:"
for i in "${array[@]}"; do
  echo -n " ${i},"
done
printf "\b \n"
