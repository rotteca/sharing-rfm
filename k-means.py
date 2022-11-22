#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


import sklearn
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import squarify
import json
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy
from scipy.stats import boxcox
import yellowbrick 
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
pd.options.display.float_format = '{:,.2f}'.format


# In[24]:


import geopandas as gpd
from bokeh.io import output_notebook, show, output_file
from bokeh.plotting import figure
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar
from bokeh.palettes import brewer
from bokeh.models import HoverTool


# In[25]:


data1 = pd.read_excel('/Users/THAO VAN/Desktop/data_rfm.xlsx',header = 0)


# In[26]:


data1.dtypes


# In[27]:


data1.describe()


# In[17]:


data1.shape


# In[28]:


data1.isnull().sum().sort_values(ascending=False)


# In[30]:


data1.dropna(inplace=True)


# In[31]:


data1.isnull().sum().sort_values(ascending=False)


# In[32]:


plt.boxplot(data1.Recency)
Q1 = data1.Recency.quantile(0.25)
Q3 = data1.Recency.quantile(0.75)
IQR = Q3 - Q1
data1 = data1[(data1.Recency >= Q1 - 1.5*IQR) & (data1.Recency <= Q3 + 1.5*IQR)]


# In[33]:


plt.boxplot(data1.Frequency)
Q1 = data1.Frequency.quantile(0.25)
Q3 = data1.Frequency.quantile(0.75)
IQR = Q3 - Q1
data1 = data1[(data1.Frequency >= Q1 - 1.5*IQR) & (data1.Frequency <= Q3 + 1.5*IQR)]


# In[34]:


plt.boxplot(data1.Monetary)
Q1 = data1.Monetary.quantile(0.25)
Q3 = data1.Monetary.quantile(0.75)
IQR = Q3 - Q1
data1 = data1[(data1.Monetary >= Q1 - 1.5*IQR) & (data1.Monetary <= Q3 + 1.5*IQR)]


# In[35]:


iqr = data1['Recency'].quantile(0.75) - data1['Recency'].quantile(0.25)
lim = np.abs((data1['Recency'] - data1['Recency'].median()) / iqr) < 2.22
# replace outliers with nan
data1.loc[:, ['Recency']] = data1.where(lim, np.nan)
data1.dropna(subset=['Recency'], inplace=True) # drop rows with NaN in numerical columns
data1.shape


# In[36]:


plt.boxplot(data1.Recency)


# In[37]:


iqr = data1['Frequency'].quantile(0.75) - data1['Frequency'].quantile(0.25)
lim = np.abs((data1['Frequency'] - data1['Frequency'].median()) / iqr) < 2.22
# replace outliers with nan
data1.loc[:, ['Frequency']] = data1.where(lim, np.nan)
data1.dropna(subset=['Frequency'], inplace=True) # drop rows with NaN in numerical columns
data1.shape


# In[39]:


plt.boxplot(data1.Frequency)


# In[40]:


iqr = data1['Monetary'].quantile(0.75) - data1['Monetary'].quantile(0.25)
lim = np.abs((data1['Monetary'] - data1['Monetary'].median()) / iqr) < 2.22
# replace outliers with nan
data1.loc[:, ['Monetary']] = data1.where(lim, np.nan)
data1.dropna(subset=['Monetary'], inplace=True) # drop rows with NaN in numerical columns
data1.shape


# In[42]:


plt.boxplot(data1.Monetary)


# In[47]:


data1['cubr_Recency']= np.cbrt(data1['Recency'])
data1['cubr_Frequency']= np.cbrt(data1['Frequency'])
data1['cubr_Monetary']= np.cbrt(data1['Monetary'])


# In[48]:


data1.describe()


# In[50]:


scaler = MinMaxScaler()


data1['minmax_Recency'] = scaler.fit_transform(data1[['Recency']])
data1['minmax_Frequency'] = scaler.fit_transform(data1[['Frequency']])
data1['minmax_Monetary'] = scaler.fit_transform(data1[['Monetary']])


# In[54]:


scaler = StandardScaler()

# fit_transform
data1['std_Recency'] = scaler.fit_transform(data1[['Recency']])
data1['std_Frequency'] = scaler.fit_transform(data1[['Frequency']])
data1['std_Monetary'] = scaler.fit_transform(data1[['Monetary']])


# In[57]:


data1['log_Recency']= np.log(data1[['Recency']])
data1['log_Frequency']= np.log(data1[['Frequency']])
data1['log_Monetary']= np.log(data1[['Monetary']])


# In[58]:


data1.agg(['skew','kurtosis']).transpose()


# In[60]:


data1.agg(['skew','kurtosis']).transpose()

X_normal_minmax = data1[['minmax_Recency', 'minmax_Frequency', 'minmax_Monetary']]
X_normal_std = data1[['std_Recency', 'std_Frequency', 'std_Monetary']]
X_normal_log = data1[['log_Recency', 'log_Frequency', 'log_Monetary']]

# Quick examination of elbow method to find numbers of clusters to make.
print('Elbow Method to determine the number of clusters to be formed:')
Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(X_normal_minmax)
Elbow_M.show()


# In[62]:


# Instantiate the clustering model and visualizer
model = KMeans(4, random_state=42)
visualizer = SilhouetteVisualizer(model, colors='yellowbrick')

visualizer.fit(X_normal_minmax)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure

