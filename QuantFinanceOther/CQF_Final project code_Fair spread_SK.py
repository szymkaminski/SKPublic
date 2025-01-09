# -*- coding: utf-8 -*-
#import modules
import pandas as pd
import math as math
import numpy as nmp
from scipy.stats import norm
from scipy.stats import t
import scipy.special as scp
from scipy.stats import chisquare
import pylab as pyl
import matplotlib.pyplot as plt
import timeit
t_start=timeit.default_timer()
# define parameters
# select the number of simulations
nSim=50000
# define the correlation scenario
# 1 - equity/CDS returns 
# any other correlation - write the desired correlation
# ATTENTION any correlation smaller tham -0,25 will result in a correlation matrix which is NOT positive semi-definite!
# in such case, an algorithm by Higham (2000) will be used to obtain the nearest positive semi-definite matrix
corrScenario=1
# select if the calculations should be based on CDS or Equity time series
# C for CDS, E for Equity
CDSorEquity='E'
# define the recovery rate
RR=0.4
# select if T-copula or Normal-copula
# N for normal, T for Tdist
NormOrT='T'
#download data
#CDS/Equity time series
if CDSorEquity=='C':
    TS = nmp.array(pd.read_csv("C:\Users\szymon.kaminski\Documents\CQF final project\Data\\TechStocksSeries.csv", sep='[|]', 
                  engine='python', header=0))
elif CDSorEquity=='E':
    TS = nmp.array(pd.read_csv("C:\Users\szymon.kaminski\Documents\CQF final project\Data\\equityreturnsALLCSV.csv", sep='[;]', 
                  engine='python', header=0))  
# CDS curves
CDS = nmp.array(pd.read_csv("C:\Users\szymon.kaminski\Documents\CQF final project\Data\\TechStocksCDS.csv", sep='[|]', 
                  engine='python', header=0))*20
#define functions
def Mean(TS):#mean
    return (sum(TS)/len(TS))
def RollingMean(TS,x):#rolling mean
    RM=0
    for i in range (0,x):
        RM=RM+TS[i]
    RM=RM/x
    return RM
def GaussianKernelX(Mx,Mn,ALen):#x axis of the Gaussian Kernel
    Points=Alen
    x=nmp.zeros((Alen,5))
    for k in range (0,5):
            x[:,k]=nmp.linspace(Mn[k],Mx[k],Points)          
    return x
def GaussianKernel(inp,Mx,Mn,bw,ALen,PoC):#the Gaussian Kernel - PDF or CDF
    Points=ALen
    PDF=nmp.zeros((Points,5))
    CDFb=nmp.zeros((Points,5))
    CDF=nmp.zeros((Points,5))
    x=nmp.zeros((Points,5))
    for k in range (0,5):
        x[:,k]=nmp.linspace(Mn[k],Mx[k],Points)
        #PDF
        for i in range (0,Points):
            for j in range(0,Alen):
                PDF[i,k]=PDF[i,k]+(1./math.sqrt(2.*math.pi))*math.exp(-0.5*pow(((inp[j,k]-x[i,k])/bw[k]),2))
            PDF[i,k]=PDF[i,k]/(bw[k]*Alen)
        #CDF
        for i in range (0,Points):
            CDFb[i,k]=PDF[i,k]/sum(PDF[:,k])
        CDF[:,k]=nmp.cumsum(CDFb[:,k])
        if PoC=='P':
                Res=PDF
        else:
                Res=CDF
    return Res
def tLogLikelihood(cKernel,CDFKernel):#t-distribution calibration loglikelihood function
    logc=nmp.zeros((25))
    for x in range(1,25):
        df=x
        part1=(1./math.sqrt(nmp.linalg.det(cKernel)))*((scp.gamma((df+5.)/2.))/scp.gamma(df/2.))*pow((scp.gamma(df/2.)/scp.gamma((df+1.)/2.)),5)
        for i in range(0,Alen):
            part2=pow((1+((t.ppf(CDFKernel[i,:],df).dot(nmp.linalg.inv(cKernel))).dot(t.ppf(CDFKernel[i,:],df)))/df),nmp.int((-(df+5)/2.)))
            part3=pow((1+pow(t.ppf(CDFKernel[i,0],df),2)/df),-(df+1)/2.)
            for j in range(1,5):
                part3=part3*pow((1+pow(t.ppf(CDFKernel[i,j],df),2)/df),-(df+1)/2.)
            logc[x]=math.log(part1*part2/part3)
    return logc
def interp(DF,x):#loglinear interpolation
    LowerBound=nmp.int(x)
    UpperBound=nmp.int(x)+1
    interp=math.exp(math.log(DF[UpperBound,1])*(x-LowerBound)/(UpperBound-LowerBound)+math.log(DF[LowerBound,1])*(UpperBound-x)/(UpperBound-LowerBound))
    return interp
def Cholesky(CM):#5x5 Cholesky matrix
    Chl=nmp.zeros((5,5))
    Chl[0,0]=1
    for i in range(1,5):
        Chl[i,0]=CM[i,0]
    Chl[1,1]=math.sqrt(1.-pow(Chl[1,0],2))
    for i in range(2,5):
        Chl[i,1]=(CM[i,1]-Chl[i,0]*Chl[1,0])/Chl[1,1]
    Chl[2,2]=math.sqrt(CM[2,2]-pow(Chl[2,0],2)-pow(Chl[2,1],2))
    for i in range(3,5):
        Chl[i,2]=(CM[i,2]-Chl[2,0]*Chl[i,0]-Chl[2,1]*Chl[i,1])/Chl[2,2]
    Chl[3,3]=math.sqrt(CM[3,3]-pow(Chl[3,0],2)-pow(Chl[3,1],2)-pow(Chl[3,2],2))
    Chl[4,3]=(CM[4,3]-Chl[3,0]*Chl[4,0]-Chl[3,1]*Chl[4,1]-Chl[3,2]*Chl[4,2])/Chl[3,3]
    Chl[4,4]=math.sqrt(CM[4,4]-pow(Chl[4,0],2)-pow(Chl[4,1],2)-pow(Chl[4,2],2)-pow(Chl[4,3],2))
    return Chl
def DefaultYearandFrac(U,CumHR,HR,Surv):#get the exact default time. ATTENTION!!!: U is already the -log(1-U) function!!!
    DY=nmp.zeros((5,3))
    for i in range(0,5):#first column of the function will be the default year
        if U[i]<CumHR[0,i]:
            DY[i,0]=0#.0
        elif U[i]<CumHR[1,i]:
            DY[i,0]=1#.0
        elif U[i]<CumHR[2,i]:
            DY[i,0]=2#.0
        elif U[i]<CumHR[3,i]:
            DY[i,0]=3#.0
        elif U[i]<CumHR[4,i]:
            DY[i,0]=4#.0
        else:
            DY[i,0]=999#.0
    for j in range(0,5):#second column of the function will be the year fraction
        if DY[j,0]==999:#.0:#exact default time equals to 0 if there is no default
            DY[j,1]=0
        else:
            DY[j,1]=-(1./HR[nmp.int(DY[j,0]),j])*math.log((1.-(1.-math.exp(-U[j])))/(Surv[nmp.int(DY[j,0]),j]))
            DY[j,2]=Surv[nmp.int(DY[j,0]),j]#3rd column will contain the survival probability
    return DY  
# find the neares positive semidefinite matrix by Nick Higham (2000)
def getAplus(A):
    eigval, eigvec = nmp.linalg.eig(A)
    Q = nmp.matrix(eigvec)
    xdiag = nmp.matrix(nmp.diag(nmp.maximum(eigval, 0)))
    return Q*xdiag*Q.T
def getPs(A, W=None):
    W05 = nmp.matrix(W**.5)
    return  W05.I * getAplus(W05 * A * W05) * W05.I
def getPu(A, W=None):
    Aret = nmp.array(A.copy())
    Aret[W > 0] = nmp.array(W)[W > 0]
    return nmp.matrix(Aret)
def nearPD(A, nit=10):
    n = A.shape[0]
    W = nmp.identity(n) 
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = getPu(Xk, W=W)
    return Yk
#time series part
Alen=len(TS)-1
Acols=len(TS[1])-1
ARet=nmp.zeros((Alen,Acols))
M=nmp.zeros(Acols)
STD=nmp.zeros(Acols)
Mn=nmp.zeros(Acols)
Mx=nmp.zeros(Acols)
BandWiths=nmp.zeros(Acols)
#calculate log returns
for y in range(0,Acols):
    for x in range(0,Alen):      
        ARet[x,y]=math.log(TS[x+1,y+1]/TS[x,y+1])
#calculate statistics
for y in range(0,Acols):
    M[y]=Mean(ARet[:,y])
    STD[y]=nmp.std(ARet[:,y])
    Mn[y]=min(ARet[:,y])
    Mx[y]=max(ARet[:,y])
    BandWiths[y]=STD[y]*pow(Alen,-1./5)
#standardize returns
ARet_Stnd=nmp.zeros((Alen,Acols))
for y in range(0,Acols):
    for x in range(0,Alen):      
        ARet_Stnd[x,y]=(ARet[x,y]-M[y])/STD[y]
#convert standardized returns to uniform numbers
ARet_U=nmp.zeros((Alen,Acols))
for y in range(0,Acols):
    for x in range(0,Alen):      
        ARet_U[x,y]=norm.cdf(ARet_Stnd[x,y])
#calculate the kernel
F=GaussianKernelX(Mx,Mn,Alen)
G=GaussianKernel(ARet,Mx,Mn,BandWiths,Alen,'C')
H=GaussianKernel(ARet,Mx,Mn,BandWiths,Alen,'P')
#numerical integratKeion over the Kernel CDF
CDFKernel=nmp.zeros((Alen,Acols))
for k in range(0,5):
    for j in range(0,Alen):    
        i=0
        while F[i,k]<ARet[j,k]:
            i+=1
        CDFKernel[j,k]=G[i,k]
for i in range(0,Alen):
    for j in range(0,5):
        if CDFKernel[i,j]>0.9999:
            CDFKernel[i,j]=0.9999
#correlation matrix for observed returns
cRet=nmp.corrcoef([ARet[:,0],ARet[:,1],ARet[:,2],ARet[:,3],ARet[:,4]])
cRetU=nmp.corrcoef([ARet_U[:,0],ARet_U[:,1],ARet_U[:,2],ARet_U[:,3],ARet_U[:,4]])
#correlation matrix for returns mapped with CDF kernel
cKernel=nmp.zeros((5,5))
if corrScenario==1:
    cKernel=nmp.corrcoef([CDFKernel[:,0],CDFKernel[:,1],CDFKernel[:,2],CDFKernel[:,3],CDFKernel[:,4]])  
else:
    for i in range (0,5):
        for j in range (0,5):
            if i==j:
                cKernel[i,j]=1.
            else:
                cKernel[i,j]=corrScenario
    cKernel=nearPD(cKernel)
ChlKernel=Cholesky(cKernel)
# T-Copula calibration
if NormOrT=='T':
    DegreesofFreedom=tLogLikelihood(cKernel,CDFKernel)
    m=0
    while DegreesofFreedom[m]<max(DegreesofFreedom):
        m=m+1
    df=m+1
    df=float(df)
#credit part
DF=nmp.zeros((6,2))
#discount curve - USD 3M tenor
DF[0,1]=1
DF[1,1]= 0.9809818 
DF[2,1]= 0.9588993 
DF[3,1]= 0.9364605 
DF[4,1]= 0.9141327 
DF[5,1]= 0.8918039 
for i in range(1,6):   
    DF[i,0]=DF[i-1,0]+1
#bootstrap survival probabilities
Surv=nmp.zeros((6,5))
for k in range (0,5):
    Surv[0,k]=1
    Surv[1,k]=(1-RR)/(1-RR+CDS[0,k])
for k in range(0,5):
    for i in range(2,6):
        for j in range (1,i):
            Surv[i,k]=Surv[i,k]+DF[j,1]*((1-RR)*Surv[j-1,k]-(1-RR+CDS[i-1,k])*Surv[j,k])/(DF[i,1]*(1-RR+CDS[i-1,k]))
        Surv[i,k]=Surv[i,k]+Surv[i-1,k]*(1-RR)/(1-RR+CDS[i-1,k])
#calcualte hazard rates
Haz=nmp.zeros((5,5))
HazCum=nmp.zeros((5,5))
for k in range (0,5):
    for i in range (0,5):
        Haz[i,k]=-math.log(Surv[i+1,k]/Surv[i,k])
for k in range (0,5):
    HazCum[:,k]=nmp.cumsum(Haz[:,k])
# declare vectors for the copula
Z=nmp.zeros((nSim,5))
ChiSq=nmp.zeros((nSim,5))
T=nmp.zeros((nSim,5))
Zcorr=nmp.zeros((nSim,5))
ZcorrU=nmp.zeros((nSim,5))
lnZCorrU=nmp.zeros((nSim,5))
DYear=nmp.zeros((nSim,5))
DTime=nmp.zeros((nSim,5))
DHist=nmp.zeros((nSim,5))
DHistSorted=nmp.zeros((nSim,5))
DYearSorted=nmp.zeros((nSim,5))
DefLegSorted=nmp.zeros((nSim,5))
DefLegReSorted=nmp.zeros((nSim,5))
NoD=nmp.zeros((nSim,5))
NoDCum=nmp.zeros((nSim,5))
DefLeg=nmp.zeros((nSim,3))
PremLeg=nmp.zeros((nSim,3))
spreadit=nmp.zeros((nSim,3))
t_start_core=timeit.default_timer()
#start the copula
for i in range(0,nSim):
    #generate random numbers
    for j in range(0,5):
        Z[i,j]=nmp.random.standard_normal()
        #student t random numbers
        if NormOrT=='T':
            ChiSq[i,j]=nmp.random.chisquare(df)
            T[i,j]=Z[i,j]/(math.sqrt(ChiSq[i,j]/df))
    #correlate random numbers with Cholesky
    #convert them to a uniform distribution
    if NormOrT=='N':
        Zcorr[i,:]=ChlKernel.dot(Z[i,:])
        ZcorrU[i,:]=norm.cdf(Zcorr[i,:])
    elif NormOrT=='T':
        Zcorr[i,:]=ChlKernel.dot(T[i,:])
        ZcorrU[i,:]=t.cdf(Zcorr[i,:],df)
    # calculate -ln(1-u)
    for l in range(0,5):
        lnZCorrU[i,l]=-math.log(1-ZcorrU[i,l])
    #calculate default time
    x=DefaultYearandFrac(lnZCorrU[i,:],HazCum,Haz,Surv)
    for k in range(0,5):
        #calculate default year
        DYear[i,k]=x[k,0]
        #calculate default time
        DTime[i,k]=x[k,1]
        #calculate exact default time
        if DYear[i,k]==999:
            DHist[i,k]=-1+DTime[i,k]
        else:
            DHist[i,k]=DYear[i,k]+DTime[i,k]
    #sort the results
    DHistSorted[i,:]=sorted(DHist[i,:], key=float)
    DYearSorted[i,:]=sorted(DYear[i,:],key=float)
    # calculation of the DEFAULT LEG
    for m in range(0,5):
        if DHistSorted[i,m]==-1:
            DefLegSorted[i,m]=0
        else:
            DefLegSorted[i,m]=(1-RR)*interp(DF,DHistSorted[i,m])*0.2
    DefLegReSorted[i,:]=sorted(DefLegSorted[i,:],key=float,reverse=True)
    DefLeg[i,0]=DefLegReSorted[i,0]
    DefLeg[i,1]=DefLegReSorted[i,1]+DefLegReSorted[i,2]
    DefLeg[i,2]=DefLegReSorted[i,3]+DefLegReSorted[i,4]
    #Calculation of the PREMIUM LEG
    NoD[i,0]=sum(1 for x in DYear[i,:] if x==0)
    NoD[i,1]=sum(1 for x in DYear[i,:] if x==1)
    NoD[i,2]=sum(1 for x in DYear[i,:] if x==2)
    NoD[i,3]=sum(1 for x in DYear[i,:] if x==3)
    NoD[i,4]=sum(1 for x in DYear[i,:] if x==4)
    NoDCum[i,:]=nmp.cumsum(NoD[i,:])
    PremLeg[i,0]=DF[0,1]
    PremLeg[i,1]=DF[0,1]
    PremLeg[i,2]=DF[0,1]
    for j in range (1,5):
        #FIRST TRANCHE
        if NoDCum[i,j]<1:
            PremLeg[i,0]=PremLeg[i,0]+DF[j,1]*((5.-NoDCum[i,j-1])/5.)
        #SECOND TRANCHE
        if NoDCum[i,j]<3:
            PremLeg[i,1]=PremLeg[i,1]+DF[j,1]*((5.-NoDCum[i,j-1])/5.)
        #THIRD TRANCHE
        if NoDCum[i,j]<5:
            PremLeg[i,2]=PremLeg[i,2]+DF[j,1]*((5.-NoDCum[i,j-1])/5.)
#calculates the rolling spread
# for i in range (1,nSim):
#     for x in range(0,3):
#         spreadit[i,x]=RollingMean(DefLeg[:,x],i)/RollingMean(PremLeg[:,x],i)
spread=nmp.zeros((3))
for x in range(0,3):
    spread[x]=Mean(DefLeg[:,x])/Mean(PremLeg[:,x])
t_end=timeit.default_timer()
cost=t_end-t_start
cost_core=t_end-t_start_core
print 'The fair spread for the first tranche is equal to %s basis points' %(spread[0]*10000.)
print 'The fair spread for the second tranche is equal to %s basis points' %(spread[1]*10000.)
print 'The fair spread for the third tranche is equal to %s basis points' %(spread[2]*10000.)
print 'The core of the code runs for %s seconds' %(cost_core)
print 'The whole code runs for %s seconds\nCongratulations!' %(cost)
#print spread

# pyl.plot(H)
# plt.hist(CDFKernel[:,2], bins = 15, cumulative=False, color='g')
# plt.title('Intel - Histogram of returns mapped to U(0,1) with Kernel CDF', fontsize = 12)
# plt.hist(ARet_U[:,2], bins = 15, cumulative=False, color='g')
# plt.title('Intel - Histogram of returns mapped to U(0,1) with naive normal CDF', fontsize = 12)



# plt.plot(H)
# plt.legend(['IBM','Oracle','Intel','Apple','Microsoft'])

# plt.plot(spreadit[:,2])
# plt.title('Spread convergence - 3rd tranche')