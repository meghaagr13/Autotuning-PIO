import json
import shlex
import subprocess
import logging
from pprint import pprint
import re
import os
import glob
import sys, getopt

logging.basicConfig(filename='app.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s',level=logging.DEBUG)


benchmarkFolder = './genericio'
outputFolder = './output'
benchmark = 'Generic-IO'
specific_commands="1024 2"
mpi_hints=""
nodes = "2"
ppn = "8"
write = "0"
try:
    opts, args = getopt.getopt(sys.argv[1:],"h:b:o:f:c:n:p:w:",["benchmark=","outputFolder=","benchmarkFolder=", "specificCommands=","nodes=","ppn=","write="])
except getopt.GetoptError:
    print('read_config_general.py -b <benchmark> -o <outputFolder> -f <benchmarkFolder> -c <specificCommands> -n <nodes> -p <ppn> -w <write>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('read_config_general.py -b <benchmark> -o <outputFolder> -f <benchmarkFolder> -c <specificCommands> -n <nodes> -p <ppn> -w <write>')
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
    elif opt in ("-w", "--write"):
        write = arg
os.chdir(benchmarkFolder)

out=subprocess.Popen(["lfs", "setstripe","-c1", "-s1048576","output/"], shell=False, stdout=subprocess.PIPE)

mpi_hints_l=""
try:
    os.remove("romhint")
except OSError:
    pass

logging.debug("MPI parameters:" + mpi_hints)
logging.debug(benchmark + " :" + specific_commands)
log = open('../generic_default.txt','a')
out=subprocess.Popen(["./run.sh", nodes, ppn, mpi_hints, "1", specific_commands], shell=False, stdout=subprocess.PIPE)
output=out.stdout.read()
logging.debug(output.decode('utf-8'))
#print(output.decode('utf-8'))
out2=subprocess.Popen(["./run.sh", nodes, ppn, mpi_hints, "0", specific_commands], shell=False, stdout=subprocess.PIPE)
output2=out2.stdout.read()
logging.debug(output2.decode('utf-8'))

#print(output2.decode('utf-8'))
#Bandwidth
bandwidth_write=re.findall("read/write bandwidth\s*:\s*([0-9]+\.[0-9]+)",output.decode('utf-8')) 
time_write=re.findall("Time for read/write\s*:\s*([0-9]+\.[0-9]+)",output.decode('utf-8')) 
readD_write=re.findall("total read/write amount\s*:\s*([0-9]+)",output.decode('utf-8')) 

bandwidth_read=re.findall("read/write bandwidth\s*:\s*([0-9]+\.[0-9]+)",output2.decode('utf-8')) 
time_read=re.findall("Time for read/write\s*:\s*([0-9]+\.[0-9]+)",output2.decode('utf-8')) 
readD_read=re.findall("total read/write amount\s*:\s*([0-9]+)",output2.decode('utf-8')) 
print(benchmark ,end = " ")
print(benchmark ,end = " ", file=log)
print(re.sub(r"\s","-",specific_commands), end = " ")
print(re.sub(r"\s","-",specific_commands), end = " ", file=log)

print("{0} {1} {2} {3} {4} {5} {6} {7}".format(write,nodes,bandwidth_write[-1],time_write[-1],readD_write[-1], bandwidth_read[0], time_read[0], readD_read[0]), end = " ")
print("{0} {1} {2} {3} {4} {5} {6} {7}".format(write,nodes,bandwidth_write[-1],time_write[-1],readD_write[-1], bandwidth_read[0], time_read[0], readD_read[0]), end = " ",file=log)

for fl in glob.glob("./output/out*"):
    os.remove(fl)


print()
print(file=log)
