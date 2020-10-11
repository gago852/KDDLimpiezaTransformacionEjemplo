# integrantes: gabriel gomez, eduardo de la hoz, stephania de la hoz

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg as la
import sys
from numpy import genfromtxt
from collections import Counter


def loader(fileName):
    return pd.read_csv(fileName)


def cleaner(df, k):
    points = np.array(df[(abs(df['x']-np.mean(df['x'])) <= k*np.std(df['x']))
                         & (abs(df['y']-np.mean(df['y'])) <= k*np.std(df['y']))])
    return points


def getOutladers(df, k):
    points = np.array(df[(abs(df['x']-np.mean(df['x'])) > k*np.std(df['x']))
                         | (abs(df['y']-np.mean(df['y'])) > k*np.std(df['y']))])
    return points


def lineap(eu1, eu2, x0, y0, x):
    return ((eu2*x)-(eu2*x0)+(eu1*y0))/eu1


def linea(v, cent):
    t = 1

    print('v '+str(v))
    print('centroide')
    print(cent)
    # p=[(t*v[0])+cent[0],(t*v[1])+cent[1]]
    p2 = [(t*v[0]), (t*v[1])]
    #print('p '+str(p))
    print('p2 '+str(p2))
    x1 = 0
    x2 = p2[0]
    y1 = 0
    y2 = p2[1]
    m = (y2-y1)/(x2-x1)
    b = y1-(m*x1)
    L = np.zeros((2, 10))
    for i in range(10):
        L[0, i] = i-4
        L[1, i] = (i-4)+b
    return L


def pc2(v):
    ecu = ecuacion(v)
    b = ecu[1]
    m = -1/ecu[0]
    L = np.zeros((2, 10))
    for i in range(10):
        L[0, i] = (i - 5)
        L[1, i] = m*(i - 5) + b
    return L


def ecuacion(v):
    t = 1
    P = [t*v[0], t*v[1]]

    x1 = 0
    x2 = -P[0]
    y1 = 0
    y2 = P[1]
    m = (y2 - y1)/(x2 - x1)
    b = y1 - (m*x1)
    L = [m, b]
    return L


k = 2.6
df = loader('2DOutliers.csv')
rawData = np.array(df)
cleanData = cleaner(df, k)
outliders = getOutladers(df, k)
print('rawdata')
print(rawData)
print('cleandata')
print(cleanData)
print('outladers')
print(outliders)
print('\n')

array_xvo = outliders[:, 0]
array_yvo = outliders[:, 1]
array_xcl = cleanData[:, 0]
array_ycl = cleanData[:, 1]


my_csv = genfromtxt('2DOutliers.csv', delimiter=',')

# Slicing array
array_flavanoids = my_csv[:, 0]

# Slicing array
array_colorintensity = my_csv[:, 1]

# Scatter plot function
colors = outliders

# Creating Panda DataFrame with Labels for Outlier Detection
outlier_df = pd.DataFrame(rawData)

print(type(outlier_df))

# rawdata
print('rawdata eugenvector')
covmatRaw = np.cov(rawData.T)
print('la matrix de covariansa es:')
print(covmatRaw)
print('\n')


resultsRaw = la.eig(covmatRaw)
print('eigenvaulues: ')
print(resultsRaw[0].real)
print('\n')


print('eigenvector: ')
print(resultsRaw[1])
print('\n')


print('varianza acumulada: ')
print(resultsRaw[0].real[0]/sum(resultsRaw[0].real))
print(resultsRaw[0].real[1]/sum(resultsRaw[0].real))

print('\n')
# cleandata
print('cleandata eugenvector')
covmatClean = np.cov(cleanData.T)
print('la matriz de covariansa es:')
print(covmatClean)
print('\n')


resultsClean = la.eig(covmatClean)
print('eigenvaulues: ')
print(resultsClean[0].real)
print('\n')


print('eigenvector: ')
print(resultsClean[1])
print('\n')


print('varianza acumulada: ')
print(resultsClean[0].real[0]/sum(resultsClean[0].real))
print(resultsClean[0].real[1]/sum(resultsClean[0].real))
print('\n')

centroideRaw = np.array([np.mean(rawData[:, 0]), np.mean(rawData[:, 1])])
print(centroideRaw)
print('\n')


centroideClean = np.array([np.mean(cleanData[:, 0]), np.mean(cleanData[:, 1])])
print(centroideClean)


# Exporting this DataFrame to CSV
# outlier_df[cleanData].to_csv("dbscan-outliersV.2.csv")

norm_flav = np.zeros(len(array_flavanoids))
norm_color = np.zeros(len(array_colorintensity))
norm_xvo = np.zeros(len(array_xvo))
norm_yvo = np.zeros(len(array_yvo))
norm_xcl = np.zeros(len(array_xcl))
norm_ycl = np.zeros(len(array_ycl))
print('lent '+str(len(array_xcl)))
for i in range(len(array_flavanoids)):
    norm_flav[i] = array_flavanoids[i]-centroideRaw[0]
for i in range(len(array_colorintensity)):
    norm_color[i] = array_colorintensity[i]-centroideRaw[1]
for i in range(len(array_xvo)):
    norm_xvo[i] = array_xvo[i]-centroideRaw[0]
for i in range(len(array_yvo)):
    norm_yvo[i] = array_yvo[i]-centroideRaw[1]
for i in range(len(array_xcl)):
    norm_xcl[i] = array_xcl[i]-centroideClean[0]
for i in range(len(array_ycl)):
    norm_ycl[i] = array_ycl[i]-centroideClean[1]

plt.subplot(221)
plt.scatter(norm_flav, norm_color, marker='o')
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.title('DATOS SIN PROCESAR CON OUTLIERS MARCADOS', fontsize=10)
plt.scatter(norm_xvo, norm_yvo, marker='o')
plt.axhline(0, color="black")
plt.axvline(0, color="black")


# plt.show()
plt.subplot(222)
plt.scatter(norm_xcl, norm_ycl, marker='o')
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.title('DATOS LIMPIOS SIN OUTLIERS', fontsize=10)
plt.axhline(0, color="black")
plt.axvline(0, color="black")
# plt.show()
plt.subplot(223)
plt.scatter(norm_flav, norm_color, marker='o')
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.title('DATOS SIN PROCESAR CON OUTLIERS MARCADOS REGRESION L', fontsize=10)
plt.scatter(norm_xvo, norm_yvo, marker='o')
plt.axhline(0, color="black")
plt.axvline(0, color="black")

eugenvectorsRaw = np.array(resultsRaw[1])
print('eugenvecRaw')
print(eugenvectorsRaw[0, :])
#x = range(-2, 8)

euRaw = np.array(eugenvectorsRaw[0, :])
print('euraw')
print(euRaw)

line1 = linea(euRaw, centroideRaw)

plt.plot(line1[0], line1[1], color='red')
euRaw2 = np.array(eugenvectorsRaw[1, :])
print('euraw')
print(euRaw2)
line2 = pc2(euRaw2)
plt.plot(line2[0], line2[1], 'orange')
# plt.show()
plt.subplot(224)
plt.scatter(norm_xcl, norm_ycl, marker='o')
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.title('DATOS LIMPIOS SIN OUTLIERS, REGRESION L', fontsize=10)
plt.axhline(0, color="black")
plt.axvline(0, color="black")

eugenvectorsClean = np.array(resultsClean[1])

euclean = np.array(eugenvectorsClean[0, :])
print('euraw')
print(euclean)

line1 = linea(euclean, centroideClean)

plt.plot(line1[0], line1[1], color='red')
euclean2 = np.array(eugenvectorsClean[1, :])
print('euraw')
print(euclean2)
line2 = pc2(euclean2)
plt.plot(line2[0], line2[1], 'orange')
# print(eugenvectorsClean)

print('\n')
print('para -2 en rawData es: ' + str(lineap(eugenvectorsRaw[0, 0],
                                             eugenvectorsRaw[0, 1], centroideRaw[0], centroideRaw[1], -2)))
print('\n')
print('para -2 en cleanData es: ' + str(lineap(eugenvectorsClean[0, 0],
                                               eugenvectorsClean[0, 1], centroideClean[0], centroideClean[1], -2)))
print('\n')

print('para 7 en rawData es: ' + str(lineap(eugenvectorsRaw[0, 0],
                                            eugenvectorsRaw[0, 1], centroideRaw[0], centroideRaw[1], 7)))
print('\n')
print('para 7 en cleanData es: ' + str(lineap(eugenvectorsClean[0, 0],
                                              eugenvectorsClean[0, 1], centroideClean[0], centroideClean[1], 7)))

plt.show()

sys.exit()
