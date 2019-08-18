#!/bin/bash
for i in {1..2}
do
	out=$(python3 default_ior.py -n 8 -c"-t 4194304 -b 104857600")
	echo $out
	echo $out >> 'default_ior.txt'
	echo "done"
	out=$(python3 default_ior.py -n 8 -c"-t 2097152  -b 104857600")
	echo $out
	echo $out >> 'default_ior.txt'
	echo "done"
	out=$(python3 default_ior.py -n 8 -c"-t 1048576 -b 104857600")
	echo $out
	echo $out >> 'default_ior.txt'
	echo "done"
	out=$(python3 default_ior.py -n 8 -c"-t 524288 -b 104857600")
	echo $out
	echo $out >> 'default_ior.txt'
	echo "done"
	out=$(python3 default_ior.py -n 8 -c"-t 262144 -b 104857600")
	echo $out
	echo $out >> 'default_ior.txt'
	echo "done"
	out=$(python3 default_ior.py -n 8 -c"-t 131072 -b 104857600")
	echo $out
	echo $out >> 'default_ior.txt'
	echo "done"
done

