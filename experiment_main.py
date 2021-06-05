# -*- coding: utf-8 -*-

# Imports
"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/drive/MyDrive/GGS_experiment/'

from ggs import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

!ls VTT_SCOTT_IAQ/room00

"""# Data Preparation"""

room12_THP_CO2 = pd.read_csv('VTT_SCOTT_IAQ/room12/room12_THP-CO2_700_20190101-20191231.csv')

room12_THP_CO2.devicetimestamp = pd.to_datetime(room12_THP_CO2.devicetimestamp, unit='s')

room12_THP_CO2.devicetimestamp.dtype

room12_THP_CO2

room12_THP_CO2 = room12_THP_CO2.set_index('devicetimestamp')

loc1 = room12_THP_CO2.index.get_loc(pd.to_datetime("2019-04-05"), method='nearest')
loc2 = room12_THP_CO2.index.get_loc(pd.to_datetime("2019-04-06"), method='nearest')

room12_THP_CO2.iloc[ loc1:loc2]

room12_THP_CO2_20190405 = room12_THP_CO2.iloc[ loc1:loc2]
room12_THP_CO2_20190405

"""# next day"""

loc1 = room12_THP_CO2.index.get_loc(pd.to_datetime("2019-07-11"), method='nearest')
loc2 = room12_THP_CO2.index.get_loc(pd.to_datetime("2019-07-12"), method='nearest')

room12_THP_CO2.iloc[ loc1:loc2]

room12_THP_CO2_20191114 = room12_THP_CO2.iloc[ loc1:loc2]
room12_THP_CO2_20190405

data = room12_THP_CO2_20191114.to_numpy()
# Select Temperature, Humidity, CO2
data = data.T
feats = [4]

data.shape

# Find 10 breakpoints at lambda = 1e-4
bps, objectives = GGS(data, Kmax = 24, lamb = 1e-4, features = feats)

print(bps)

# Find means and covariances of the segments, given the selected breakpoints
bp10 = bps[24] # Get breakpoints for K = 10
meancovs = GGSMeanCov(data, breakpoints = bp10, lamb = 1e-4, features = feats)


print("Breakpoints are at", bps)

# Plot objective vs. number of breakpoints
plotVals = range(len(objectives))
plt.plot(plotVals, objectives, 'or-')
plt.xlabel('Number of Breakpoints')
plt.ylabel(r'$\phi(b)$')
plt.show()

room12_THP_CO2_20190405.index[[0, 278, 922, 1422]]

room12_THP_CO2_20190405.plot()

breaks = bps[2]
mcs = GGSMeanCov(data, breaks, 1e-1, features=feats)
predicted = []
varPlus = []
varMinus = []
for i in range(len(mcs)):
	for j in range(breaks[i+1]-breaks[i]):
		predicted.append(mcs[i][0]) # Estimate the mean
		varPlus.append(mcs[i][0] + math.sqrt(mcs[i][1][0])) # One standard deviation above the mean
		varMinus.append(mcs[i][0] - math.sqrt(mcs[i][1][0])) # One s.d. below

plt.plot(predicted)
plt.plot(varPlus, 'r--')
plt.plot(varMinus, 'r--')
plt.ylabel('Predicted mean (+/- 1 S.D.)')
plt.show()

predicted

"""# GGS Cycle experiment (небольшой тест на работу в цикле)

"""

days = ['20190711', '20191010', '20191211', '20191205', '20191114', '20190405']

day_points = []
for day in days:
  data = globals()['room12_THP_CO2_' + day].to_numpy()
  # Select CO2
  data = data.T
  feats = [4]
  bps, objectives = GGS(data, Kmax = 24, lamb = 1e-4, features = feats)
  print(bps[2])
  day_points.append(bps[2])

print(day_points)

day_points = (np.array(day_points)).T

day_points

day_points_norm = (day_points  - day_points.mean()) / (np.std(day_points) + 0.0001)

day_points_norm

np.std(day_points, axis=1)

"""# GGS Single Experiment (код, примененный к каждому выбранному дню)"""

data = room12_THP_CO2_20190405.to_numpy()
# Select Temperature, Humidity, CO2
data = data.T
feats = [4]

data.shape

# Find 10 breakpoints at lambda = 1e-4
bps, objectives = GGS(data, Kmax = 24, lamb = 1e-4, features = feats)

print(bps)

# Find means and covariances of the segments, given the selected breakpoints
bp10 = bps[24] # Get breakpoints for K = 10
meancovs = GGSMeanCov(data, breakpoints = bp10, lamb = 1e-4, features = feats)


print("Breakpoints are at", bps)

# Plot objective vs. number of breakpoints
plotVals = range(len(objectives))
plt.plot(plotVals, objectives, 'or-')
plt.xlabel('Number of Breakpoints')
plt.ylabel(r'$\phi(b)$')
plt.show()

room12_THP_CO2_20190405.index[[0, 278, 922, 1422]]

room12_THP_CO2_20190405.plot()

breaks = bps[2]
mcs = GGSMeanCov(data, breaks, 1e-1, features=feats)
predicted = []
varPlus = []
varMinus = []
for i in range(len(mcs)):
	for j in range(breaks[i+1]-breaks[i]):
		predicted.append(mcs[i][0]) # Estimate the mean
		varPlus.append(mcs[i][0] + math.sqrt(mcs[i][1][0])) # One standard deviation above the mean
		varMinus.append(mcs[i][0] - math.sqrt(mcs[i][1][0])) # One s.d. below

plt.plot(predicted)
plt.plot(varPlus, 'r--')
plt.plot(varMinus, 'r--')
plt.ylabel('Predicted mean (+/- 1 S.D.)')
plt.show()

predicted

"""# PIR проверка"""

room12_THP_PIR = pd.read_csv('VTT_SCOTT_IAQ/room12/room12_THP-PIR_623_20190101-20191231.csv')

room12_THP_PIR
