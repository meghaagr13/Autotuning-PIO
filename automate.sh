#!/bin/bash
rm S3D-IO/output/*
for i in `seq 1 1`
do
	python3 model/randomParameters.py 
	python3 read_config.py
done
