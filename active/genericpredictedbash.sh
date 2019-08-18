touch ../genericio/romhint
python predicted_model-generic.py 16777216 4
cp confex.json ../confex.json
cd ../
python3 generic_read_config_general.py -n4 -c"16777216 2"
