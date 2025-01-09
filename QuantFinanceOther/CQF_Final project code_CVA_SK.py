import numpy as nmp
import pandas as pd
import math as math
import pylab as pyl
import matplotlib.pyplot as plt
import timeit
t_start=timeit.default_timer()
#define functions
def Mean(TS):#mean
    return (sum(TS)/len(TS))
#downloaddata
HJM = nmp.array(pd.read_csv("\PCAoutput.csv", sep='[|]', 
                  engine='python', header=0))
StartCurve=nmp.array(pd.read_csv("\curve.csv", sep='[|]', 
                  engine='python', header=0))
CDS = nmp.array(pd.read_csv("\CDSXVAcsv.csv", sep='[|]', 
                  engine='python', header=0))
#declare variables - simulation part
dt=1./2
nSim=100000
FIX=0.01
N=1000000
nCurves=11
MtMCFs=nmp.zeros((11,11))# MtMs per cash flow
EE=nmp.zeros((nSim,11))# MTMs per data point
EPE=nmp.zeros((nSim,11))# positive MTMs per data point
AvgEPE=nmp.zeros((11))
CVAPeriod=nmp.zeros((10))
drift=HJM[:,0]
vol1=HJM[:,1]
vol2=HJM[:,2]
vol3=HJM[:,3]
t=nmp.zeros((nCurves))
t=nmp.linspace(0,5,nCurves)
Tenor=nmp.zeros((51))
Tenor=nmp.linspace(0,25,51)
SimCurve=nmp.zeros((nCurves,11))
DiscCurve=nmp.zeros((nCurves,11))
LiborOIS=nmp.array([0.00073,0.00151,0.00203,0.00195,0.00196,0.00208,0.00223,0.00237,0.00245,0.00249,0.00249])
#declare variables - credit part
RR=0.4
Surv=nmp.zeros((11))
Haz=nmp.zeros((10))
#simulation part
for k in range (0,nSim):
    for i in range (0,11):
        SimCurve[0,i]=StartCurve[i]
    # Simulate the curves with HJM
    for j in range (1,nCurves):
        for i in range (0,10):#the 11th step needs to use a backward derivative
            SimCurve[j,i]=SimCurve[j-1,i]+drift[i]*dt+(vol1[i]*nmp.random.standard_normal()+vol2[i]*nmp.random.standard_normal()+vol3[i]*nmp.random.standard_normal())*math.sqrt(dt)+dt*(SimCurve[j-1,i+1]-SimCurve[j-1,i])/(Tenor[i+1]-Tenor[i])
        SimCurve[j,10]=SimCurve[j-1,10]+drift[10]*dt+(vol1[10]*nmp.random.standard_normal()+vol2[10]*nmp.random.standard_normal()+vol3[10]*nmp.random.standard_normal())*math.sqrt(dt)+dt*(SimCurve[j-1,10]-SimCurve[j-1,9])/(Tenor[10]-Tenor[9])
    # calculate discount factors
    for j in range (0,nCurves):
        DiscCurve[j,0]=1
        for i in range (1,11):
            DiscCurve[j,i]=DiscCurve[j,i-1]/(1.+(SimCurve[j,i-1]-LiborOIS[i-1])*0.5)
    # FV of swap
    for j in range (0,11):#iterates 12 exposure points
        for i in range (0,11-j):#calculates the fair value at one exposure point
            MtMCFs[i,j]=N*(SimCurve[j,i]-FIX)*DiscCurve[j,i]*dt
        EE[k,j]=sum(MtMCFs[:,j])
#EPE matrix
for k in range(0,nSim):
    for j in range(0,11):
        EPE[k,j]=max(EE[k,j],0)
#average of EPE
for j in range(0,10):
    AvgEPE[j]=(Mean(EPE[:,j+1])+Mean(EPE[:,j]))/2.
# credit part
Surv[0]=1
Surv[1]=(1-RR)/(1-RR+0.5*CDS[0,0])
for i in range(2,11):
    for j in range (1,i):
        Surv[i]=Surv[i]+DiscCurve[0,j]*((1-RR)*Surv[j-1]-(1-RR+0.5*CDS[i-1,0])*Surv[j])/(DiscCurve[0,i]*(1-RR+0.5*CDS[i-1,0]))
    Surv[i]=Surv[i]+Surv[i-1]*(1-RR)/(1-RR+0.5*CDS[i-1,0])
for i in range (0,10):
    Haz[i]=-math.log(Surv[i+1]/Surv[i])
#CVA computation
for i in range (0,10):
    CVAPeriod[i]=AvgEPE[i]*Haz[i]*(1.-RR)*DiscCurve[0,i+1]
CVA=sum(CVAPeriod)
t_end=timeit.default_timer()
cost=t_end-t_start
print 'The CVA is equal to %s GBP' %(CVA)
print 'The code runs for %s seconds\nCongratulations!' %(cost)

# plt.plot(AvgEPE)
# plt.title('lognormal LN(0,1)', fontsize = 18)
# plt.xlabel('RV', color='k', fontsize=12)
# 
# x1=nmp.linspace(0,4.5,10)
# pyl.title('Net exposure profile of the IRS')
# pyl.xlabel('Tenor')
# pyl.ylabel('NPE')
# pyl.plot(x1,AvgEPE[1:11],'b-')
# # pyl.plot(ST,'r-')
# pyl.show()