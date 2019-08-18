import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df= pd.read_csv("generic_default.txt",delimiter=" ")
print(df.iloc[:,7])
print(df.iloc[:,4])
writeb_def=np.sum(df.iloc[:,7].values)
readb_def=np.sum(df.iloc[:,4].values)

df= pd.read_csv("genericbest.txt",delimiter=" ")
writeb=np.sum(df.iloc[:,7].values)
readb=np.sum(df.iloc[:,4].values)

print(df.iloc[:,7])
print(df.iloc[:,4])
d=readb
e=writeb


print("Fraction increment read, write, overall"+str(d/readb_def)+" "+str(e/writeb_def)+" "+str((d+e)/(readb_def+writeb_def)))



