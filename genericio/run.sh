#!/bin/bash

# remove data files
# declare arrays
echo "Format is <nodes> <ppn <mpihint> <read=0/write=1> <if write rows seed>"
# prefix of all jobs for this question
jobPrefix="Generic-IO"
node=$1   
echo "No of nodes - ${node}"
   # assign name of job
   jobName="${jobPrefix}_${node}"
   echo ${jobName}
   export commnd=$5
   export write=$4
   export romhint=$3
   # run the job
   echo ${node}, $2, $write
   if [ "$write" -eq 0 ]; then
       echo "In read benchmark...."
	   export path="./mpi/GenericIOBenchmarkRead"
       export commnd=""
       output=$(qsub -N ${jobName} -l nodes=${node}:ppn=$2 -v path -v commnd -v romhint qsub.sh)
   else
       echo "In write benchmark...."
       export path="./mpi/GenericIOBenchmarkWrite"
       output=$(qsub -N ${jobName} -l nodes=${node}:ppn=$2 -v path -v commnd -v romhint qsub.sh)       
   fi
   echo $output
   # extract id of job
   id=$( echo $output  | awk -F. '{print $1}')
   outfile=${jobName}.o${id}
   echo "Running the job"
   while [ ! -f "$outfile" ]
   do 
	continue
   done
   # append data to data file for plots
   echo "${outfile} finished" 
   out=$(cat $outfile)
   echo "$out"
   rm "$outfile"
