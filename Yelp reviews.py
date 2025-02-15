import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

yelp = pd.read_csv('yelp.csv')

yelp.head()

yelp.info()

yelp.describe()

yelp['text length'] = yelp['text'].apply(len)

g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length')