import pandas
import torch
from matplotlib import pyplot as plt
import numpy as np
import math
 
# load data
print('loading data...')
NORS = torch.load('NORSWithCategories.pt')
TWEETS = torch.load('TweetsWithCategories.pt')

# create counts
def getCounts(df):
	# initialize
	categories = {'fish':0, 'shellfish':0, 'dairy':0, 'egg':0, 'meat':0, 'poultry':0, 'fruit/nut':0, 'vegetable':0, 'oil/sugar':0, 'grain/bean':0}

	# iterate through all rows
	for index, row in df.iterrows():
		# iterate through categories
		for category in row['Category']:
			categories[category] = categories[category] + 1

	return np.array(list(categories.values()))

def getCountsNORS(df):
	# initialize
	categories = {'fish':0, 'shellfish':0, 'dairy':0, 'egg':0, 'meat':0, 'poultry':0, 'fruit/nut':0, 'vegetable':0, 'oil/sugar':0, 'grain/bean':0}

	# iterate through all rows
	for index, row in df.iterrows():
		# iterate through categories
		for category in row['Category']:
			categories[category] = categories[category] + row['EstimatedPrimary']

	return np.array(list(categories.values()))

# ---------------------------------all data---------------------------------

# Creating data
categories = ['fish', 'shellfish', 'dairy', 'egg', 'meat', 'poultry', 'fruit/nut', 'vegetable', 'oil/sugar', 'grain/bean']
 
NORScounts = getCountsNORS(NORS)
TWEETScounts = getCounts(TWEETS)

# normalize
NORS_perc = 100*(NORScounts/sum(NORScounts))
TWEETS_perc = 100*(TWEETScounts/sum(TWEETScounts))

# Creating plots
X_axis = np.arange(len(categories))
plt.bar(X_axis - 0.2, NORS_perc, 0.4, label='NORS')
plt.bar(X_axis + 0.2, TWEETS_perc, 0.4, label='Tweets')

plt.xticks(X_axis, categories)

# show plot
plt.xlabel("Categories")
plt.ylabel("Percent of Classified Foods")
plt.title("Percentage Breakdown of Food Categories")
plt.legend()
plt.show()

# calculate rmse
def mae(nors,tweets):
	Sum = 0
	count = 0
	for i in range(len(nors)):
		# absolute error
		Sum += abs(nors[i]-tweets[i])
		count += 1

	return Sum/count

print('all data mean absolute error: ', mae(NORS_perc,TWEETS_perc))

# ---------------------------------------------------------------------------
