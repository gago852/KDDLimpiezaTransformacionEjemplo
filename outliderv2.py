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


def linea(eu1, eu2, x0, y0, x):
    return ((eu2*x)-(eu2*x0)+(eu1*y0))/eu1


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

plt.subplot(221)
plt.scatter(array_flavanoids, array_colorintensity, marker='o')
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.title('DATOS CON OUTLIERS MARCADOS', fontsize=10)
plt.scatter(array_xvo, array_yvo, marker='o')

eugenvectorsRaw = np.array(resultsRaw[1])
print(eugenvectorsRaw)
x = range(-2, 8)

plt.plot(x, [linea(eugenvectorsRaw[0, 0], eugenvectorsRaw[0, 1],
                   centroideRaw[0], centroideRaw[1], i) for i in x], 'red')
plt.plot(x, [linea(eugenvectorsRaw[1, 0], eugenvectorsRaw[1, 1],
                   centroideRaw[0], centroideRaw[1], i) for i in x], 'orange')

# plt.show()
plt.subplot(222)
plt.scatter(array_flavanoids, array_colorintensity, marker='o')
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.title('DATOS EN BRUTO', fontsize=10)
# plt.show()
plt.subplot(223)
plt.scatter(array_xvo, array_yvo, marker='o')
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.title('SOLO OUTLIERS', fontsize=10)
# plt.show()
plt.subplot(224)
plt.scatter(array_xcl, array_ycl, marker='o')
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.title('DATOS LIMPIOS SIN OUTLIERS', fontsize=10)

eugenvectorsClean = np.array(resultsClean[1])
print(eugenvectorsClean)

plt.plot(x, [linea(eugenvectorsClean[0, 0], eugenvectorsClean[0, 1],
                   centroideClean[0], centroideClean[1], i) for i in x], 'red')
plt.plot(x, [linea(eugenvectorsClean[1, 0], eugenvectorsClean[1, 1],
                   centroideClean[0], centroideClean[1], i) for i in x], 'orange')

plt.show()


print('\n')

print(eugenvectorsRaw)
print('\n')
print(eugenvectorsRaw[0,0])
print(eugenvectorsRaw[1,0])

print('\n')
test=(eugenvectorsRaw[0,0]*eugenvectorsRaw[1,0])+(eugenvectorsRaw[0,1]*eugenvectorsRaw[1,1])
print(test)




#print('para -2 en rawData es: ' + str(linea(eugenvectorsRaw[0, 0],
#                                            eugenvectorsRaw[0, 1], centroideRaw[0], centroideRaw[1], -2)))
#print('\n')
#print('para -2 en cleanData es: ' + str(linea(eugenvectorsClean[0, 0],
#                                              eugenvectorsClean[0, 1], centroideClean[0], centroideClean[1], -2)))
#print('\n')

#print('para 7 en rawData es: ' + str(linea(eugenvectorsRaw[0, 0],
 #                                          eugenvectorsRaw[0, 1], centroideRaw[0], centroideRaw[1], 7)))
#print('\n')
#print('para 7 en cleanData es: ' + str(linea(eugenvectorsClean[0, 0],
#                                             eugenvectorsClean[0, 1], centroideClean[0], centroideClean[1], 7)))


sys.exit()
