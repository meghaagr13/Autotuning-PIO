import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df= pd.read_csv("default.txt",delimiter=" ")
print(df.iloc[:,5])
print(df.iloc[:,2])
writeb_def=np.sum(df.iloc[:,5].values)
readb_def=np.sum(df.iloc[:,2].values)

df= pd.read_csv("best.txt",delimiter=" ")
writeb=np.sum(df.iloc[:,5].values)
readb=np.sum(df.iloc[:,2].values)

print(df.iloc[:,5])
print(df.iloc[:,2])
d=readb
e=writeb


print("Fraction increment read, write, overall"+str(d/readb_def)+" "+str(e/writeb_def)+" "+str((d+e)/(readb_def+writeb_def)))




