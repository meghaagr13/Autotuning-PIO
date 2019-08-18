#!/bin/bash
	out=$(python3 default_ior.py -n 1 -c"-t 2621440 -b 104857600")
	echo $out
	out=$(python3 default_ior.py -n 2 -c"-t 2621440 -b 104857600")
	echo $out
	out=$(python3 default_ior.py -n 4 -c"-t 2621440 -b 104857600")
	echo $out
	out=$(python3 default_ior.py -n 8 -c"-t 2621440 -b 104857600")
	echo $out
	out=$(python3 default_ior.py -n 16 -c"-t 2621440 -b 104857600")
	echo $out

