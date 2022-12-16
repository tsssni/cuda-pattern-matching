import matplotlib.pyplot as plt
import numpy as np

class FuncRunData:
    def __init__(self,name):
        self.name=name
        self.cpuPatternLen=[]
        self.gpuPatternLen=[]
        self.cpuRuntime=[]
        self.gpuRuntime=[]
        self.speedUpPatternLen=[]
        self.speedUp=[]

    def addCpu(self,len,time):
        self.cpuPatternLen.append(len)
        self.cpuRuntime.append(time)
    
    def addGpu(self,len,time):
        self.gpuPatternLen.append(len)
        self.gpuRuntime.append(time)

    def calcSpeedUp(self):
        for i in range(0,len(self.gpuPatternLen)):
            for j in range(0,len(self.cpuPatternLen)):
                if(self.cpuPatternLen[j]==self.gpuPatternLen[i]):
                    self.speedUpPatternLen.append(self.gpuPatternLen[i])
                    self.speedUp.append(self.cpuRuntime[j]/self.gpuRuntime[i])


exp=open('exp_gene.txt')
legends=[]
funcRunDataDict=dict()

while True:
    line=exp.readline()
    if not line:
        break

    line=line.rstrip('\n')
    keys=line.split(',')

    funcName=keys[0].split(':')[1]
    patternLen=keys[1].split(':')[1]
    runtime=float(keys[2].split(':')[1])

    if ' CPU' in funcName:
        funcName=funcName.replace(' CPU','')
        if(funcName not in funcRunDataDict):
            funcRunDataDict[funcName]=FuncRunData(funcName)
            legends.append(funcName)

        funcRunDataDict[funcName].addCpu(patternLen,runtime);

    if ' GPU' in funcName:
        funcName=funcName.replace(' GPU','')
        if(funcName not in funcRunDataDict):
            funcRunDataDict[funcName]=FuncRunData(funcName)
            legends.append(funcName)

        funcRunDataDict[funcName].addGpu(patternLen,runtime);

for funcName in legends:
    funcRunDataDict[funcName].calcSpeedUp()
    
plt.figure(figsize=(20,10),dpi=100)
plt.yticks(np.arange(0,39,3))
plt.ylim((0,39))
for funcName in funcRunDataDict:
        frd=funcRunDataDict[funcName]
        plt.plot(frd.cpuPatternLen,frd.cpuRuntime,marker='o',markersize='15')
plt.legend(legends)
plt.show()

plt.figure(figsize=(20,10),dpi=100)
plt.yticks(np.arange(0,39,3))
plt.ylim((0,39))
for funcName in funcRunDataDict:
        frd=funcRunDataDict[funcName]
        plt.plot(frd.gpuPatternLen,frd.gpuRuntime,marker='o',markersize='15')
plt.legend(legends)
plt.show()

plt.figure(figsize=(20,10),dpi=100)
plt.yticks(np.arange(0,39,3))
plt.ylim((0,39))
for funcName in funcRunDataDict:
        frd=funcRunDataDict[funcName]
        plt.plot(frd.speedUpPatternLen,frd.speedUp,marker='o',markersize='15')
plt.legend(legends)
plt.show()