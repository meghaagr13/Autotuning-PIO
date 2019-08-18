import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

df= pd.read_csv("default_ior.txt",delimiter=" ",error_bad_lines=False,encoding='utf-8')
print(df.iloc[:,5].str.split('-'))
df[['col1','col2','col3','col4','col5','col6','col7','col8', 'col9' ]] = df.iloc[:,5].str.split('-',expand=True)
df['totalProcesses']= df['col8']

df[['totalProcesses']]= df[['totalProcesses']].astype(str)

df['config']=df.iloc[:,5]

df_best= pd.read_csv("iorbest.txt",delimiter=" ")
print(df_best.iloc[:,1].str.split('-',expand=True))
df_best[['col1','col2','col3','col4','col5','col6','col7','col8'  ]] = df_best.iloc[:,1].str.split('-',expand=True)
df_best['totalProcesses']= df_best['col8']
df_best[['totalProcesses']]= df_best[['totalProcesses']].astype(str)

print(df.shape)
df_best['config']=df_best.iloc[:,1]

print(df_best.shape)
df = pd.merge(df,df_best,on=['config','totalProcesses'])
print(list(df.columns.values))
#print(df.iloc[:,1]["200-200-400-4-4-4-1"])
fig, axarr = plt.subplots(10,constrained_layout=True)
i = 0
pd.set_option('display.max_columns', 500)
for f in sorted(set(df['totalProcesses'])):
    k =  df[df['totalProcesses'] == f]
    k.plot.bar(x='config',y=[6,33,9,36],title=f,figsize=(15,10),legend=False,color=('sandybrown','dodgerblue','lightcoral','darkmagenta'),fontsize=10,ax=axarr[i])
    if i % 4 == 0:
        axarr[i].legend(["read_default","read_active","write_default","write_active"])

    axarr[i].legend(["read_default","read_active","write_default","write_active"])
    axarr[i].tick_params(axis='x',labelrotation=0)
    i = i+1
plt.savefig("iordefaultvsbest.png")
