#!/bin/bash
for i in {1..2}
do
#        out=$(python3 default_ior.py -n 1 -c"-t 4194304 -b 209715200")
#	echo $out
#	echo $out >> 'default_ior.txt'
#	echo "done"
#	out=$(python3 default_ior.py -n 2 -c"-t 4194304 -b 209715200")
#	echo $out
#	echo $out >> 'default_ior.txt'
#	echo "done"
#	out=$(python3 default_ior.py -n 4 -c"-t 4194304 -b 209715200")
#	echo $out
#	echo $out >> 'default_ior.txt'
#	echo "done"
#	out=$(python3 default_ior.py -n 8 -c"-t 4194304 -b 209715200")
#	echo $out
#	echo $out >> 'default_ior.txt'
#	echo "done"
#	out=$(python3 default_ior.py -n 16 -c"-t 4194304 -b 209715200")
#	echo $out
#	echo $out >> 'default_ior.txt'
#	echo "done"
#	out=$(python3 default_ior.py -n 20 -c"-t 4194304 -b 209715200")
#	echo $out
#	echo $out >> 'default_ior.txt'
#	echo "done"
#	out=$(python3 default_ior.py -n 24 -c"-t 4194304 -b 209715200")
#	echo $out
#	echo $out >> 'default_ior.txt'
#	echo "done"
#	out=$(python3 default_ior.py -n 28 -c"-t 4194304 -b 209715200")
#	echo $out
#	echo $out >> 'default_ior.txt'
#	echo "done"
	out=$(python3 default_ior.py -n 32 -c"-t 4194304 -b 209715200")
	echo $out
	echo $out >> 'default_ior.txt'
	echo "done"
done

