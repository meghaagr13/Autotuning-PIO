import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import pandas as pd

df= pd.read_csv("best",delimiter=" ")
df['col1'] = df.iloc[:,1] #.str.split('',expand=True)
df['col2']=df['col1']
df['totalProcesses']= df.iloc[:,3]
df[['totalProcesses']]= df[['totalProcesses']].astype(int)
df['config']=df.iloc[:,1].astype(float)
df['readB'] = df.iloc[:,4]
df['writeB'] = df.iloc[:,7]
df_best= pd.read_csv("genericbest.txt",delimiter=" ")
df_best['col1'] = df_best.iloc[:,1] #.str.split('',expand=True)
df_best['col2']=df_best['col1']
df_best['totalProcesses']= df_best.iloc[:,3]
df_best[['totalProcesses']]= df_best[['totalProcesses']].astype(int)
df_best['config']=df_best.iloc[:,1].astype(float)
print(df.shape)
print(df_best.shape)
print(df)
print(df_best)
print("OKOKOK")
df = df[['config','totalProcesses','readB','writeB']].groupby(['config','totalProcesses']).mean()
df = df.reset_index()
print(df)
print("NOTOKOKOK")
df = pd.merge(df,df_best,on=['config','totalProcesses'])
print(list(df.columns.values))
df = df.sort_values(by=['config'])

print(df)
#print(df.iloc[:,1]["200-200-400-4-4-4-1"])
fig, axarr = plt.subplots(4,constrained_layout=True)
o = [0,1,2,2,3]#[4,3,0,2,1]#[3,1,0,4,2]
i = 0
nodes=[2,4,8,16,28]
pd.set_option('display.max_columns', 500)
for f in sorted(set(df['totalProcesses'].astype(int))):
    print(f)
#    if f==8:
#       continue 
# print(df[df['totalProcesses']==f])
    k =  df[df['totalProcesses'] == f]
    print(k)
    k.plot.bar(x='config',y=[2,8,3,11],figsize=(15,10),legend=False,color=('sandybrown','dodgerblue','lightcoral','darkmagenta'),fontsize=10,ax=axarr[o[i]])
    if i % 5 == 4:
        axarr[o[i]].legend(["read_default","read_active","write_default","write_active"],prop={'size': 8},loc="upper left",ncol=2)
    axarr[o[i]].tick_params(axis='x',labelrotation=0)
    axarr[o[i]].set_xlabel('')
    axarr[o[i]].set_ylabel('I/O Bandwidth')

    i = i+1
#plt.show()
plt.savefig("genericdefaultvsbest.png")
