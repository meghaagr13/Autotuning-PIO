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

benchmarkFolder = './ior-3.2.0'
outputFolder = './output'
benchmark = 'IOR'
specific_commands="-t 1024K -b 1024K"
nodes = "4"
ppn = "8"
try:
    opts, args = getopt.getopt(sys.argv[1:],"h:b:o:f:c:n:p:",["benchmark=","outputFolder=","benchmarkFolder=", "specificCommands=","nodes=","ppn="])
except getopt.GetoptError:
    print('read_config_general.py -b <benchmark> -o <outputFolder> -f <benchmarkFolder> -c <specificCommands> -n <nodes> -p <ppn>')
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
        print(specific_commands)
    elif opt in ("-n", "--nodes"):
        nodes = arg
    elif opt in ("-p", "--ppn"):
        ppn = str(arg)
#print(benchmark + specific_commands + str(nodes) + str(ppn))
os.chdir(benchmarkFolder)
out=subprocess.Popen(["lfs", "getstripe","-d", "."], shell=False, stdout=subprocess.PIPE) 
data['lfs']['setstripe']['size']="1048576"
data['lfs']['setstripe']['count']="1"
print(data)
for key in data["lfs"]:
    command = "lfs " + key + "  "
    lustreC = "" 
    d =  data["lfs"][key]
    print(d)
    if 'filename' in d:
        command += d['filename']
    else: 
        command += "."
    
    lustreC+="lustreStripeSize="+d['size']
    lustreC+=",lustreStripeCount="+d['count']
    logging.debug("Lustre parameters:"+ lustreC)

mpi_hints=""
os.remove("mpihints")
with open("mpihints",'a') as mpihint:
    for key in data["mpi"]:
        mpi_hints+=key
        mpi_hints+="="+data["mpi"][key]+";"
        mpihint.write("IOR_HINT__MPI__"+key+" "+data["mpi"][key]+"\n")

mpi_hints=""

logging.debug("MPI parameters:" + mpi_hints)
logging.debug(benchmark + " :" + specific_commands)
log = open('../iorstats.txt','a')
out=subprocess.Popen(["./run.sh", nodes, ppn, mpi_hints,specific_commands,lustreC], shell=False, stdout=subprocess.PIPE)
output=out.stdout.read()
#print(output.decode('utf-8'))
logging.debug(output.decode('utf-8'))
readValues=re.findall("read\s*([0-9]+\.[0-9]+)\s*([0-9]+\.*[0-9]*)\s*([0-9]+\.*[0-9]*)\s*([0-9]+\.[0-9]+)\s*([0-9]+\.[0-9]+)\s*([0-9]+\.[0-9]+)\s*([0-9]+\.[0-9]+)",output.decode('utf-8'))
writeValues=re.findall("write\s*([0-9]+\.[0-9]+)\s*([0-9]+\.*[0-9]*)\s*([0-9]+\.*[0-9]*)\s*([0-9]+\.[0-9]+)\s*([0-9]+\.[0-9]+)\s*([0-9]+\.[0-9]+)\s*([0-9]+\.[0-9]+)",output.decode('utf-8'))
#print(readValues)
#Bandwidth
readB=readValues[0][0]
writeB=writeValues[0][0]

#Data
readD = readValues[0][1]
writeD = writeValues[0][1]

#time
readT=readValues[0][4]
writeT=writeValues[0][4]

openT=float(readValues[0][3]) + float(writeValues[0][3])
closeT=float(readValues[0][5]) + float(writeValues[0][5])

print(benchmark ,end = " ")
print(benchmark ,end = " ", file=log)
print(re.sub(r"\s","-",specific_commands)+"-n-"+str(nodes), end = " ")
print(re.sub(r"\s","-",specific_commands)+"-n-"+str(nodes), end = " ", file=log)

print("{0} {1} {2} {3} {4} {5} {6} {7}".format(readB,readD,readT,writeB,writeD,writeT,openT,closeT), end = " ")
print("{0} {1} {2} {3} {4} {5} {6} {7}".format(readB,readD,readT,writeB,writeD,writeT,openT,closeT), end = " ",file=log)

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
