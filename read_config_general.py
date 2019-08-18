import json
import shlex
import subprocess
import logging
from pprint import pprint
import re
import os
import sys, getopt

#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(filename='app.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s',level=logging.DEBUG)
#logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

with open('confex.json') as f:
    data = json.load(f)

benchmarkFolder = './S3D-IO'
outputFolder = './output'
benchmark = 'S3D-IO'
specific_commands="50 50 100 2 2 4 1"
nodes = "2"
ppn = "8"
system=""
try:
    opts, args = getopt.getopt(sys.argv[1:],"h:b:o:f:c:n:p:s:",["benchmark=","outputFolder=","benchmarkFolder=", "specificCommands=","nodes=","ppn=","system="])
except getopt.GetoptError:
    print('read_config_general.py -b <benchmark> -o <outputFolder> -f <benchmarkFolder> -c <specificCommands> -n <nodes> -p <ppn> -s <system>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('read_config_general.py -b <benchmark> -o <outputFolder> -f <benchmarkFolder> -c <specificCommands> -n <nodes> -p <ppn>')
        sys.exit()
    elif opt in ("-b", "--benchmark"):
        benchmark = arg
    elif opt in ("-f", "--benchmarkFolder"):
        benchmarkFolder = arg
    elif opt in ("-o", "--outputFolder"):
        outputFolder = arg
    elif opt in ("-c", "--specificCommands"):
        specific_commands = arg
    elif opt in ("-n", "--nodes"):
        nodes = arg
    elif opt in ("-p", "--ppn"):
        ppn = str(arg)
    elif opt in ("-s", "--system"):
        system = arg
#print(benchmark + specific_commands + str(nodes) + str(ppn))
os.chdir(benchmarkFolder)
#out=subprocess.Popen(["lfs", "getstripe","-d", "."], shell=False, stdout=subprocess.PIPE) 
benchmark = system+benchmark
for key in data["lfs"]:
    command = "lfs " + key + "  "
    d =  data["lfs"][key]
    if 'filename' in d:
        command += d['filename']
    else: 
        command += "."
    if 'size' in d:
        command+= " -s " + str(d['size'])
    if 'count' in d:
        command+= " -c " + str(d['count'])
    #print(str(command))
    logging.debug("Lustre parameters:"+ command)
    os.chdir(outputFolder)
    subprocess.Popen(shlex.split(str(command)), shell=False)

mpi_hints=""
os.chdir('../')
for key in data["mpi"]:
    mpi_hints+=key
    mpi_hints+="="+data["mpi"][key]+";"


logging.debug("MPI parameters:" + mpi_hints)
logging.debug(benchmark + " :" + specific_commands)
log = open('../stats.txt','a')
out=subprocess.Popen(["./run.sh", nodes, ppn, mpi_hints,specific_commands], shell=False, stdout=subprocess.PIPE)
output=out.stdout.read()
print(output.decode('utf-8'))
logging.debug(output.decode('utf-8'))
#Bandwidth
readB=re.findall("read  bandwidth\s*:\s*([0-9]+\.[0-9]+)",output.decode('utf-8'))
writeB=re.findall("write bandwidth\s*:\s*([0-9]+\.[0-9]+)",output.decode('utf-8'))

#Data
readD=re.findall("total read  amount\s*:\s*([0-9]+\.[0-9]+)",output.decode('utf-8'))
writeD=re.findall("total write amount\s*:\s*([0-9]+\.[0-9]+)",output.decode('utf-8'))

#time
readT=re.findall("Time for read\s*:\s*([0-9]+\.[0-9]+)",output.decode('utf-8'))
writeT=re.findall("Time for write\s*:\s*([0-9]+\.[0-9]+)",output.decode('utf-8'))
openT=re.findall("Time for open\s*:\s*([0-9]+\.[0-9]+)",output.decode('utf-8'))
closeT=re.findall("Time for close\s*:\s*([0-9]+\.[0-9]+)",output.decode('utf-8'))

print(benchmark ,end = " ")
print(benchmark ,end = " ", file=log)
print(re.sub(r"\s","-",specific_commands), end = " ")
print(re.sub(r"\s","-",specific_commands), end = " ", file=log)

print("{0} {1} {2} {3} {4} {5} {6} {7}".format(readB[0],readD[0],readT[0],writeB[0],writeD[0],writeT[0],openT[0],closeT[0]), end = " ")
print("{0} {1} {2} {3} {4} {5} {6} {7}".format(readB[0],readD[0],readT[0],writeB[0],writeD[0],writeT[0],openT[0],closeT[0]), end = " ",file=log)

for hints in data["lfs"]["setstripe"]:
        print(data["lfs"]["setstripe"][hints],end=" ")
        print(data["lfs"]["setstripe"][hints],end=" ",file=log)

hints_array=mpi_hints.split(";")
for hints in hints_array:
    if(len(hints.split('=')) == 2): 
        print(hints.split("=")[1],end=" ")
        print(hints.split("=")[1],end=" ",file=log)

print()
print(file=log)
