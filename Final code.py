#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
import os
import math
from scipy.special import gamma
from scipy import integrate
from sklearn.preprocessing import StandardScaler
import scipy.stats












# Aggregate All Function
def ALL(Mean7_Delay,e17, e27, e37, e47,df):
    param,param1,seasonT,EseasonT,Likespring,Likesummer,Likefall,Likewinter,Like3p,param_primespring,param1_primespring,param_primesummer,param1_primesummer,param_primefall,param1_primefall,param_primewinter,param1_primewinter,Rspring,Rsummer,Rfall,Rwinter= parameters_Eseason(Mean7_Delay,e17, e27, e37, e47,df)
    Result=pd.DataFrame([[int(df['Segment ID'].unique()[0]),e17]],columns=['Segment ID','type'])
    A=dictionary.loc[dictionary['Segment ID']==int(df['Segment ID'].unique()[0])]
    Result['Length']=A['Segment Length(Kilometers)'].unique()[0]
    '''
    These function temporary deleted
    
    INS1M2W=(inverseS1_model2('winter',EseasonT,param_primewinter,param1_primewinter))
    INS1M2SM=inverseS1_model2('summer',EseasonT,param_primesummer,param1_primesummer)
    INS1M2F=inverseS1_model2('fall',EseasonT,param_primefall,param1_primefall)
    INS1M2SP=inverseS1_model2('spring',EseasonT,param_primespring,param1_primespring)
    EseasonTEX,SeasonTEX,scenario2=sce2(EseasonT,seasonT) 
    scenario2_model2=inverseS2_model2(EseasonTEX,SeasonTEX,scenario2)
    scenario2=inverseS2(EseasonTEX,SeasonTEX,scenario2)
    INS1M1=inverseS1(EseasonT,param,param1)
    LIKDF=pd.DataFrame([0])
    LIKM2W=(Likelihood_Model2('winter',EseasonT,param_primewinter,param1_primewinter))
    LIKM2SM=Likelihood_Model2('summer',EseasonT,param_primesummer,param1_primesummer)
    LIKM2F=Likelihood_Model2('fall',EseasonT,param_primefall,param1_primefall)
    LIKM2SP=Likelihood_Model2('spring',EseasonT,param_primespring,param1_primespring)
    '''
    LIKDF['Log_CGEV_M2_spring']=LIKM2SP
    LIKDF['Log_CGEV_M2_summer']=LIKM2SM
    LIKDF['Log_CGEV_M2_fall']=LIKM2F
    LIKDF['Log_CGEV_M2_winter']=LIKM2W
    
    sm=0
    for i in list([LIKM2SP,LIKM2SM,LIKM2F,LIKM2W]):
        
        if np.isnan(i):
            sm=sm+np.nan_to_num(i)
        else:
            sm=sm+i
    LIKDF['Log_CGEV_M2_totsl']=sm 
    LIKDF=LIKDF.drop(columns=[0])
    
    LIK=Likelihood(EseasonT,'Like12name','Like48name',param,param1)

    Result=pd.concat([Result,INS1M1,INS1M2SP,INS1M2SM,INS1M2F,INS1M2W],axis=1)
    
    Result['RL_GEV_SP']=Rspring
    Result['RL_GEV_SM']=Rsummer
    Result['RL_GEV_F']=Rfall
    Result['RL_GEV_W']=Rwinter
    Result=pd.concat([Result,scenario2,scenario2_model2,LIKDF],axis=1)
    
    Result['Log_CGEV_4season']=LIK['Like48name'].unique().sum()
    Result['Log_CGEV_20season']=LIK['Like12name'].sum()
    Result = Result.loc[:,~Result.columns.duplicated()]
    

    Result['Log_GEV_W']=Likewinter
    Result['Log_GEV_SP']=Likespring
    Result['Log_GEV_SM']=Likesummer
    Result['Log_GEV_F']=Likefall
    Result['Log_GEV_3P']=Like3p
    return Result





def sce2(EseasonT,seasonT):
    List=list(EseasonT.columns)
    List.remove('Year')
    List.remove('season')
    List.remove('num')
    List.remove('Mean')
    List.remove('avg')
    List1=list(seasonT.columns)
    List1.remove('Year')
    List1.remove('season')
    List1.remove('num')
    List1.remove('Mean')
    
    scenario2=pd.DataFrame(np.array([[0,0,0,0]]),columns=['springmax','summermax','fallmax','wintermax'])
    listSeason=pd.Series(EseasonT['season'].unique())
    listSeason=list(listSeason.dropna())
    for i in listSeason:
        df=EseasonT.loc[EseasonT['season']==i]
        df1=seasonT.loc[seasonT['season']==i]
        
        listyear=df['Year'].unique()
        listyear1=df1['Year'].unique()
        df=df.loc[df['Year']==listyear.max()]
        df1=df1.loc[df1['Year']==listyear1.max()]
        index=df.loc[df['Year']==listyear.max()].index[0]
        index1=df1.loc[df1['Year']==listyear1.max()].index[0]
        df2=df[List]
        max2=np.array(df2.max(axis=1)*df['Mean'])
        scenario2[list(df['season'].unique())[0]+'max']=max2
        EseasonT=EseasonT.drop([index])
        seasonT=seasonT.drop([index1])
        EseasonT=EseasonT.dropna(thresh=10)
    return EseasonT,seasonT ,scenario2
    
    






def season1(name,df1,Meantype):
    cond=1
    for i in range(3,12):
        df1=df1[df1.Month != i]
    df1['Month'] =df1['Month'].map({12 : 0, 1 :1 , 2:2})
    yearU=df.Year.unique()
    yearU1=yearU-1
    dfT=pd.DataFrame([])
    for i , j in zip(yearU, yearU1):
        
        df21=df1.loc[(df1['Year']== i) & (df1['Month']!=0)]
        
        df21=df21.append(df1.loc[(df['Year']== j)& (df1['Month']==0)])
        dfF=pd.DataFrame(columns=['Year','season','Mean','STD','e1','e2','e3'])
        dfF=dfF.append([0,0,0,0,0,0,0])
        df21=df21[df21[Meantype]>0]
        if df21.Year.count() ==3:
            cond=0
            dfF['Year']=i
            dfF['season']=str(name)
            dfF['Mean']=df21[Meantype].mean()
            dfF['STD']=df21[Meantype].std()
            cont=range(df21.Year.count())
            dfF['e1']=(df21[Meantype].iloc[cont[0]])/(df21[Meantype].mean())
            dfF['e2']=(df21[Meantype].iloc[cont[1]])/(df21[Meantype].mean())
            dfF['e3']=(df21[Meantype].iloc[cont[2]])/(df21[Meantype].mean())
        elif df21.Year.count() ==2:
            cond=0
            dfF['Year']=i
            dfF['season']=str(name)
            dfF['Mean']=df21[Meantype].mean()
            dfF['STD']=df21[Meantype].std()
            cont=range(df21.Year.count())
            dfF['e1']=(df21[Meantype].iloc[cont[0]])/(df21[Meantype].mean())
            dfF['e2']=(df21[Meantype].iloc[cont[1]])/(df21[Meantype].mean())
            dfF['e3']=None
        elif df21.Year.count() ==1:
            cond=1
        else:
            continue
        if cond==0:
            dfT=dfT.append(dfF)
            
        
    dfT=dfT.drop_duplicates(inplace=False)
    return dfT















def season (name,a,b,c,df1,Meantype):
    dfT=pd.DataFrame([])
    df2=df1.loc[(df1['Month']== a) | (df1['Month']==(b)) | (df1['Month']==(c)) ]
    yearU=df2.Year.unique()
    for i in yearU:    
        df21=df2.loc[(df2['Year']== i)]
        dfF=pd.DataFrame(columns=['Year','season','Mean','STD','e1','e2','e3','num'])
        dfF=dfF.append([0,0,0,0,0,0,0])
        
        df21=df21[df21[Meantype]>0]
        if df21.Year.count()==3:
            cond=0
            dfF['Year']=i
            dfF['season']=str(name)
            dfF['Mean']=df21[Meantype].mean()
            dfF['STD']=df21[Meantype].std()
            cont=range(df21.Year.count())
            dfF['e1']=(df21[Meantype].iloc[cont[0]])/(df21[Meantype].mean())
            dfF['e2']=(df21[Meantype].iloc[cont[1]])/(df21[Meantype].mean())
            dfF['e3']=(df21[Meantype].iloc[cont[2]])/(df21[Meantype].mean())
            
        elif df21.Year.count() ==2:
            cond=0
            dfF['Year']=i
            dfF['season']=str(name)
            dfF['Mean']=df21[Meantype].mean()
            dfF['STD']=df21[Meantype].std()
            cont=range(df21.Year.count())
            dfF['e1']=(df21[Meantype].iloc[cont[0]])/(df21[Meantype].mean())
            dfF['e2']=(df21[Meantype].iloc[cont[1]])/(df21[Meantype].mean())
            dfF['e3']=None
            
        elif df21.Year.count() ==1:
            cond=1
        else:
            continue
        if cond==0:
            dfT=dfT.append(dfF)
            
        
    dfT=dfT.drop_duplicates(inplace=False)
    return dfT






def Eseason (name,a,b,c,df1,Meantype,error1,error2,error3,error4):
    dfT=pd.DataFrame([])
    dfF=pd.DataFrame([])
    df2=df1.loc[(df1['Month']== a) | (df1['Month']==(b)) | (df1['Month']==(c)) ]
    
    yearU=df2.Year.unique()
    for i in yearU:    
        df21=df2.loc[(df2['Year']== i)]
        df21=df21[df21[Meantype]>0]
        
        df31=df21[[error1,error2,error3,error4]]
        df31=df31[df31>0]
        
        df31=df31.to_numpy().reshape(1,-1)
        df31= df31[~np.isnan(df31)]
        dfF=pd.DataFrame([df31])

        if df21.Year.count() ==3:
            cond=0
            dfF['Year']=i
            dfF['season']=str(name)
            dfF['Mean']=df21[Meantype].mean()
            dfF['num']=df21.Year.count()


        elif df21.Year.count() ==2:
            cond=0
            dfF['Year']=i
            dfF['season']=str(name)
            dfF['Mean']=df21[Meantype].mean()
            dfF['num']=df21.Year.count()


        elif df21.Year.count() ==1:
            cond=1
            dfF['num']=1
        else:
            continue
        if cond==0:
            dfT=dfT.append(dfF)
        if df21.Year.count() >1:
            sum1=dfT['num']*dfT['Mean']
            dfT['avg']=sum1.sum()/(dfT['num'].sum())
        

    
    
    return dfT



def Eseason1(name,df1,Meantype,error1,error2,error3,error4):
    cond=1
    for i in range(3,12):
        df1=df1[df1.Month != i]
    df1['Month'] =df1['Month'].map({12 : 0, 1 :1 , 2:2})
    
    yearU=df.Year.unique()
    yearU1=yearU-1
    dfT=pd.DataFrame([])
    for i , j in zip(yearU, yearU1):
        
        df21=df1.loc[(df1['Year']== i) & (df1['Month']!=0)]
        
        
        df21=df21.append(df1.loc[(df['Year']== j)& (df1['Month']==0)])
        df21=df21[df21[Meantype]>0]
        df31=df21[[error1,error2,error3,error4]]
        df31=df31[df31>0]
        
        df31=df31.to_numpy().reshape(1,-1)
        df31= df31[~np.isnan(df31)]
        dfF=pd.DataFrame([df31])
        
        
        if df21.Year.count() ==3:
            cond=0
            dfF['Year']=i
            dfF['season']=str(name)
            dfF['Mean']=df21[Meantype].mean()
            dfF['num']=df21.Year.count()

        elif df21.Year.count() ==2:
            cond=0
            dfF['Year']=i
            dfF['season']=str(name)
            dfF['Mean']=df21[Meantype].mean()
            dfF['num']=df21.Year.count()
  
        elif df21.Year.count() ==1:
            cond=1
            dfF['num']=df21.Year.count()
        else:
            continue
        if cond==0:
            dfT=dfT.append(dfF)
        
        if df21.Year.count() >1:
            sum1=dfT['num']*dfT['Mean']
            dfT['avg']=sum1.sum()/(dfT['num'].sum())
        
    
       
    
    return dfT





def ListC(EseasonT):
    List=list(EseasonT.columns)
    List.remove('Year')
    List.remove('season')
    List.remove('num')
    List.remove('Mean')
    List.remove('avg')
    return List
# function for fitting parameter and calculate likelihood
def LikelihoodGEV(dfFit):
    LLH=0
    y=dfFit.to_numpy()
    y=y.reshape(-1,1)
    dfFit=pd.DataFrame(y)
    dfFit=dfFit[dfFit>0]
    dfFit=dfFit.dropna()
    y=dfFit.to_numpy()
    if y.shape[0]>2:
        dist = getattr(scipy.stats, 'genextreme')
        param_prime2 = dist.fit(y)
        LLH = dist.logpdf(y,*param_prime2).sum()
        return LLH,param_prime2
    else:
        return np.nan,np.nan
    

# function for fitting parameter 
def parameterSeasonal(dfseason):
    df11=dfseason.reset_index(inplace = False)
    dfFit=pd.DataFrame([])
    dfFit=dfFit.append([df11['e1'] , df11['e2'] ,df11['e3']])
    y=dfFit.to_numpy()
    y=y.reshape(-1,1)
    dfFit=pd.DataFrame(y)
    dfFit=dfFit[dfFit>0]
    dfFit=dfFit.dropna()
    y=dfFit.to_numpy()
    if y.shape[0]>2:
        dist = getattr(scipy.stats, 'genextreme')
        param_prime2 = dist.fit(y)
        LLH = dist.logpdf(y,*param_prime2).sum()
        return param_prime2
    else:
        return np.nan
    

def returnvalue(prameter,delta_avg):
    
    pr=prameter
    if np.nan_to_num(pr[0])==0:
        
        return 'notdefine'
    else:
        dist = getattr(scipy.stats, 'genextreme')
        ret = (pr[1]-(pr[2]/pr[0])*(1-(-np.log(.916667))**(-pr[0])))*delta_avg
    return ret

    

# In[ ]:



dictionary=pd.read_csv('C:/Users/LabPC/Desktop/dictionary\\dictionary.csv')
Z=get_all_file_paths ('C:\\Users\\LabPC\\Desktop\\Final process\\1')
def get_all_file_paths(directory):
    file_paths = []

    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths

s=-1
for file in Z:
    
    
    df=pd.read_csv(file)
    if df['Slope7'].unique()[0]>0:
        s=s+1
        if s==0:
            a1=ALL('Mean7 Delay','e17', 'e27', 'e37', 'e47',df)
        else:
            a1=a1.append(ALL('Mean7 Delay','e17', 'e27', 'e37', 'e47',df))
        a1=a1.append(ALL('Mean7 Delayreftravel','eR17', 'eR27', 'eR37', 'eR47',df))
        
    else:
        s=s+1
        if s==0:
            a1=ALL('Mean7 Delayreftravel','eR17', 'eR27', 'eR37', 'eR47',df)
        else:
            a1=a1.append(ALL('Mean7 Delayreftravel','eR17', 'eR27', 'eR37', 'eR47',df))
a1.to_csv('1.csv')




