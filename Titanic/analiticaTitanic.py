"""
Algo más serio, vamos a por el titanic

Pagina guía: https://www.kaggle.com/c/titanic
        == Data Dictionary ==
        =====================
        Variable	Definition	Key
        survival	Survival	0 = No, 1 = Yes
        pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
        sex	Sex
        Age	Age in years
        sibsp	# of siblings / spouses aboard the Titanic
        parch	# of parents / children aboard the Titanic
        ticket	Ticket number
        fare	Passenger fare
        cabin	Cabin number
        embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

        Jupyter: py -m notebook

"""
import numpy as np
import matplotlib.pyplot as plt
#from sklearn
import pandas as pd
import math
from scipy.stats import itemfreq

from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

print("Titanic v0.1")
df = pd.read_csv( "train.csv",
    names=["PassengerId","Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"])
####################################################
MiArrayGeneral =df.values    #Convertimos en numpy array un daaFrame de Pandas

####################################### 1 .- Ordenamos con Pandas el Array de trabajo
# Objetivo Decition tree con 'n' fearutes () y un label (binario, >50 o < 50k€)

        #tipos de selecion por columnas en Pandas
pieces = df["Sex"]
pieces2 = df.iloc[3:5,0:2]
pieces3 = df["Survived"]

pi_A=pd.DataFrame(pieces)  #convertir a PANDA dataFrame
pi_B=pd.DataFrame(pieces3)
features=pd.concat([pi_B, pi_A] , axis=1)  #fabrico features Matrix de 16000x2
#print(features)
labels = pd.DataFrame(df["Survived"])     #fabrico el label matri
#print (labels)

######################################  2.- Convertimos a numPY para hace MachineLearning
np_Labels   = labels.values
np_Features = features.values



################################# ETL convetir array de string en numero para trabajar con numeros (columnas Education y labels)
#                                           luego se puede deshacer la conversion.
le1 = preprocessing.LabelEncoder()
le1.fit(np.ravel(np_Labels))
labels_Num = (le1.transform(np.ravel(np_Labels)))   # el ravel, convierte columna en fila y parece que le gusta más
#print("Strigs: ", np_Labels)
#print("Numeros: ",labels_Num)


################################ Pintamos Sencillo
#df['Sex'].value_counts().plot(kind='bar', color=['b','r'], alpha=0,5)
#plt.show()
valor=df['Sex'].value_counts().plot(kind='bar')
plt.show()
valor=df['Survived'].value_counts().plot(kind='bar')
print (valor)
plt.show()
df.plot(kind='scatter', x='Pclass',y='Age')
plt.show()
#Ahora in plot filtrando (molaaaa)
df[df['Survived']==1]['Age'].value_counts().plot(kind='bar')
plt.show()

# hacemos una agrupacion por edades
grupos=[0,10,20,30,40,50,60,70,80]
df['AgeGrouped'] = pd.cut(df['Age'],grupos)  #creamos una nueva columna
df[df['Survived']==1]['AgeGrouped'].value_counts().plot(kind='bar')

plt.show()



""" J3

#************************************** Pintar scatter
############################ ScatterPlot
    # las 20 primeras filas columna 0 para la x, 12 para la Y
     #plt.scatter(MiArrayGeneral[0:20 , 0:1], MiArrayGeneral[0:20 , 12:13], c="g", alpha=0.5,
#            label="J3")
rico = np.ma.masked_equal(labels_Num, 0.)
pobre = np.ma.masked_where(labels_Num < 1.  , labels_Num)

#print(rico)
#print(pobre)
muestras =16000
rico_X=np.arange(muestras)   #len(labels_Num))
rico_y=np.arange(muestras)   #len(labels_Num))
pobre_X=np.arange(muestras)  #len(labels_Num))
pobre_y=np.arange(muestras)  #len(labels_Num))

indiceRico=indicePobre=0
for i in range(muestras):
    if(labels_Num[i]==1):
        rico_X[indiceRico] = pieces[i]
        rico_y[indiceRico] = pieces3[i]
        indiceRico+=1
        #print("Rico",rico_X[i], rico_y[i])
    else:
        pobre_X[indicePobre] = pieces[i]
        pobre_y[indicePobre] = pieces3[i]
        indicePobre+=1
        #print("Pobre",pobre_X[i], pobre_y[i])

print(indiceRico,indicePobre)
print(rico_X[0:indiceRico])
print(rico_y[0:indiceRico])
print(pobre_X[0:indicePobre])
print(pobre_y[0:indicePobre])
print(labels_Num[:muestras])
plt.scatter(rico_X[0:indiceRico], rico_y[0:indiceRico],  marker='o', c="blue")
plt.scatter(pobre_X[:indicePobre], pobre_y[:indicePobre],alpha=0.5,  marker='^', c="red")


#plt.scatter(pieces, pieces3, c="r", alpha=0.5,
#            marker ='*', label="Age/education > Salary")
plt.xlabel("fnlwgt")
plt.ylabel("educacion")
plt.legend(loc=2)
plt.show()



##########################################  4.- dividimos el dataser en training y test de manera randon
X_train, X_test, y_train, y_test = train_test_split(
    np_Features, labels_Num, test_size=0.33, random_state=42)

print("Strigs: ", np_Labels)
print("Numeros: ",labels_Num)

################################################ AT LAST!!! some MachineLearning
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)


########################################################Pintar el Decision tree
## Esto genera un fichero .dot que se puede ver en http://www.webgraphviz.com/ con solo copiar el texto en la pantalla
import graphviz
from graphviz import Graph
dot_data = tree.export_graphviz(clf, out_file='tree.dot')
#graph = graphviz.Source(dot_data)
#graph.render("USA population2")

J3"""

"""  Material extra... codigo de aprendizaje, para reutilizar.

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
"""
