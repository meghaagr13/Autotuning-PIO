

#out=$(python3 default_S3D.py -n 16 -c"300 300 600 4 4 8 1")
#echo $out
#echo $out >> 'default.txt'
#echo "done"
rm S3D-IO/output/*    

out=$(python3 default_S3D.py -n 8 -c"300 300 300 4 4 4 1")
echo $out
echo $out >> 'default.txt'
echo "done"
rm S3D-IO/output/*    
out=$(python3 default_S3D.py -n 4 -c"300 300 150 4 4 2 1")
echo $out
echo $out >> 'default.txt'
echo "done"
rm S3D-IO/output/*    
out=$(python3 default_S3D.py -n 2 -c"150 150 300 2 2 4 1")
echo $out
echo $out >> 'default.txt'
echo "done"
rm S3D-IO/output/*    

