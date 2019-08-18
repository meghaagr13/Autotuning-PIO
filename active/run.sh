for i in {1..4}
do
out=$(python generic\ active\ learning.py -n 28 -c"104857600 2")
echo $out >> best_generic.txt
echo $out 
out=$(python generic\ active\ learning.py -n 28 -c"1073741824 2")
echo $out >> best_generic.txt
echo $out 
out=$(python generic\ active\ learning.py -n 28 -c"13107200 2")
echo $out >> best_generic.txt
echo $out 
out=$(python generic\ active\ learning.py -n 28 -c"209715200 2")
echo $out >> best_generic.txt
echo $out 
out=$(python generic\ active\ learning.py -n 28 -c"26214400 2")
echo $out >> best_generic.txt
echo $out 
out=$(python generic\ active\ learning.py -n 28 -c"268435456 2")
echo $out >> best_generic.txt
echo $out 
out=$(python generic\ active\ learning.py -n 28 -c"419430400 2")
echo $out >> best_generic.txt
echo $out 
out=$(python generic\ active\ learning.py -n 28 -c"52428800 2")
echo $out >> best_generic.txt
echo $out 
out=$(python generic\ active\ learning.py -n 28 -c"536870912 2")
echo $out >> best_generic.txt
echo $out 
out=$(python generic\ active\ learning.py -n 28 -c"104857600 2")
echo $out >> best_generic.txt
echo $out 
out=$(python generic\ active\ learning.py -n 16 -c"1073741824 2")
echo $out >> best_generic.txt
echo $out 
out=$(python generic\ active\ learning.py -n 16 -c"13107200 2")
echo $out >> best_generic.txt
echo $out 
out=$(python generic\ active\ learning.py -n 16 -c"209715200 2")
echo $out >> best_generic.txt
echo $out 
out=$(python generic\ active\ learning.py -n 16 -c"26214400 2")
echo $out >> best_generic.txt
echo $out 
out=$(python generic\ active\ learning.py -n 16 -c"268435456 2")
echo $out >> best_generic.txt
echo $out 
out=$(python generic\ active\ learning.py -n 16 -c"419430400 2")
echo $out >> best_generic.txt
echo $out 
out=$(python generic\ active\ learning.py -n 16 -c"52428800 2")
echo $out >> best_generic.txt
echo $out 
out=$(python generic\ active\ learning.py -n 16 -c"536870912 2")
echo $out >> best_generic.txt
echo $out 
done

