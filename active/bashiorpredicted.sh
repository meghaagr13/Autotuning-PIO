python predicted_model-ior.py 2621440 104857600 20
cp confex.json ../
cd ../
python3 ior_read_config_general.py -n20 -c"-t 2621440 -b 104857600"
cd active/
python predicted_model-ior.py 2621440 104857600 24
cp confex.json ../
cd ../
python3 ior_read_config_general.py -n24 -c"-t 2621440 -b 104857600"
cd active/
python predicted_model-ior.py 2621440 104857600 28
cp confex.json ../
cd ../
python3 ior_read_config_general.py -n28 -c"-t 2621440 -b 104857600"
cd active/
python predicted_model-ior.py 2621440 104857600 32
cp confex.json ../
cd ../
python3 ior_read_config_general.py -n32 -c"-t 2621440 -b 104857600"
cd active/
