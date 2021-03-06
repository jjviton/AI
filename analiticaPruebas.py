"""
Primera prueba handsOn de Data Sciencest

Pagina guía: http://scg.sdsu.edu/dataset-adult_r/
datasets
https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
https://www.valentinmihov.com/2015/04/17/adult-income-data-set/

"""
import numpy as np
import matplotlib.pyplot as plt
#from sklearn
import pandas as pd
import math
from scipy.stats import itemfreq

print("empezamos bien?")
df = pd.read_csv( "adult.test",
    names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

MiArrayGeneral =df.values    #Convertimos numpy array

#print (df.head())
#print(df.values)

############################ ScatterPlot
    # las 20 primeras filas columna 0 para la x, 12 para la Y
#plt.scatter(MiArrayGeneral[0:20 , 0:1], MiArrayGeneral[0:20 , 12:13], c="g", alpha=0.5,
#            label="J3")

plt.scatter(MiArrayGeneral[0:16000 , 0:1], MiArrayGeneral[0:16000 , 12:13], c="r", alpha=0.5,
            marker ='*', label="J3")
plt.xlabel("age")
plt.ylabel("hour_per_Week")
plt.legend(loc=2)
#plt.show()



####################################### Dsitribuciones  numpy.histogram(y, bins=y)

plt.rcdefaults()
fig, ax = plt.subplots()

estatus = np.array(MiArrayGeneral[0:16000 , 5:6])
#estatusT = estatus.T
#print(estatusT)
posiblesStatus = np.array(['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed','Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])

# Contamos el numero de ocurrencias por tipo de datos
cuenta1= itemfreq(estatus)
print(cuenta1)

#numero de datos distintos
y_pos = np.arange(len(cuenta1[0:60 , 0:1]))
#Pinto las barras
ax.barh(y_pos, cuenta1[0:60 , 1:2], align='center',
        color='blue', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(cuenta1[0:60 , 0:1])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('number of cases')
ax.set_title('Status')
#plt.show()

############################################################################
#######################################  Graficamos Educacion versus Sexo

fig1, ax1 = plt.subplots()

mivar= 88
_jj=0
education = np.array(MiArrayGeneral[0:16000 , 3:4])  #slicing columna de Educacion [3]

## Separamos Men Women
Mujeres  =  np.empty([16000], dtype=object)  #definirlo como array d sentences
Hombres  =  np.empty([16000], dtype=object)
print (education.dtype)

for i in range(len(education)):
    #Mujeres[i]= MiArrayGeneral[0:1:]
    #print(MiArrayGeneral[i][9])
    if(MiArrayGeneral[i][9] == "Female"):
        #np.put(Mujeres, i, MiArrayGeneral[i][3])
        Mujeres[_jj]=MiArrayGeneral[i][3]   #  [i:(i+1) , 3:4]
        _jj +=1
        #print(MiArrayGeneral[i:(i+1) , 3:4])
print("Mujeres",Mujeres)

_jj=0
for i in range(len(education)):
    #Mujeres[i]= MiArrayGeneral[0:1:]
    #print(MiArrayGeneral[i][9])
    if(MiArrayGeneral[i][9] == "Male"):
        #np.put(Mujeres, i, MiArrayGeneral[i][3])
        Hombres[_jj]=MiArrayGeneral[i][3]   #  [i:(i+1) , 3:4]
        _jj +=1
        #print(MiArrayGeneral[i:(i+1) , 3:4])
print("hombres", Hombres)


age = np.array(MiArrayGeneral[0:16000 , 0:1])           #slicing columna de edad
# Contamos el numero de ocurrencias por tipo de datos
# Como nos ordena?
educationOrdered = itemfreq(education)
educationOrdeMuj = itemfreq(Mujeres)
educationOrdeHombres = itemfreq(Hombres)

print(educationOrdered)
print("hombre",educationOrdeHombres)
print("mujeres",educationOrdeMuj)

#numero de datos distintos
y_pos = np.arange(len(educationOrdeMuj[0:16000 , 0:1]))
#Pinto las barras
width = 0.35
ax1.barh(y_pos, educationOrdeMuj[0:16000 , 1:2], width, align='center',
        color='pink', ecolor='black')
ax1.barh(y_pos+width, educationOrdeHombres[0:16000 , 1:2], width, align='center',
        color='green', ecolor='black')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(educationOrdered[0:60 , 0:1])
ax1.invert_yaxis()  # labels read top-to-bottom
ax1.set_xlabel('Cantidad')
ax1.set_title('Education Hombre/Mujeres')
plt.show()
