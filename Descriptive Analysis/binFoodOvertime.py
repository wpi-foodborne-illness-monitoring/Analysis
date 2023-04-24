import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

# options: 'fish', 'shellfish', 'dairy', 'egg', 'meat', 'poultry', 'fruit/nut', 'vegetable', 'oil/sugar', 'grain/bean'
SearchCategory = 'poultry'

# load data
print('loading data...')
NORS = torch.load('NORSWithCategories.pt')
TWEETS = torch.load('TweetsWithCategories.pt')

# filter by food
NORS = NORS[pd.DataFrame(NORS.Category.tolist()).isin([SearchCategory]).any(1).values]
TWEETS = TWEETS[pd.DataFrame(TWEETS.Category.tolist()).isin([SearchCategory]).any(1).values]

TWEETS.index = pd.to_datetime(TWEETS['created_at'])
NORS.index = pd.to_datetime(NORS['DateFirstIll'])

# filter the data to be a comparible timeframe
TWEETS = TWEETS[TWEETS.index.year < 2022]
TWEETS = TWEETS[TWEETS.index.year > 2016]

NORS = NORS[NORS.index.year > 2016]

# filter out iwp
TWEETS = TWEETS[TWEETS['username'] != 'iwaspoisoned_']

# -----------------------------------------------

TWEETS = TWEETS.groupby(by=[TWEETS.index.year, TWEETS.index.month])['created_at'].count()

TWEETS = TWEETS.to_frame()
listOfTweetDates = list(TWEETS.index.values)

NORS = NORS.groupby(by=[NORS.index.year, NORS.index.month])['EstimatedPrimary'].sum()

NORS = NORS.to_frame()
listOfNORSDates = list(NORS.index.values)

# -------------------------create dictionary-------------------------

allDatesSet = set()

for dateTuple in listOfTweetDates:
	allDatesSet.add(dateTuple)

for dateTuple in listOfNORSDates:
	allDatesSet.add(dateTuple)

listAllDates = sorted(list(allDatesSet))

tweetDateDict = {}
NORSDateDict = {}

for dateTuple in listAllDates:
	tweetDateDict[dateTuple] = 0
	NORSDateDict[dateTuple] = 0

# add all tweet counts
for i in range(len(listOfTweetDates)):
	tweetDateDict[listOfTweetDates[i]] = tweetDateDict[listOfTweetDates[i]] + int(TWEETS.iloc[i])

# add all NORS counts
for i in range(len(listOfNORSDates)):
	NORSDateDict[listOfNORSDates[i]] = NORSDateDict[listOfNORSDates[i]] + int(NORS.iloc[i])

# ----------------------graph dates----------------------

# Creating data
 
NORScounts = np.array(list(NORSDateDict.values()))
TWEETScounts = np.array(list(tweetDateDict.values()))

# normalize
NORS_perc = 100*(NORScounts/sum(NORScounts))
TWEETS_perc = 100*(TWEETScounts/sum(TWEETScounts))

# Creating plots
X_axis = np.arange(len(listAllDates))
plt.bar(X_axis - 0.2, NORS_perc, 0.4, label='NORS')
plt.bar(X_axis + 0.2, TWEETS_perc, 0.4, label='Tweets')

plt.xticks(X_axis, listAllDates)

# show plot
plt.xlabel("Date")
plt.ylabel("Percent")
plt.title("Percent of " + SearchCategory + " Overtime")
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

# -------------------------correlation analysis with Lags-------------------------

# make a big numpy array --> df
npArray = np.hstack((np.array(listAllDates),TWEETS_perc.reshape(-1,1)))
npArray = np.hstack((npArray,NORS_perc.reshape(-1,1)))

df = pd.DataFrame(npArray, columns = ['year','month','tweetFreq','norsFreq'])


from scipy import signal
def ccf_values(series1, series2):
    p = series1
    q = series2
    p = (p - np.mean(p)) / (np.std(p) * len(p))
    q = (q - np.mean(q)) / (np.std(q))  
    c = np.correlate(p, q, 'full')
    return c
    
ccf_caseFreq = ccf_values(df['tweetFreq'], df['norsFreq'])

lags = signal.correlation_lags(len(df['tweetFreq']), len(df['norsFreq']))
print(max(ccf_caseFreq[:75]), lags[list(ccf_caseFreq).index(max(ccf_caseFreq[:75]))])

def ccf_plot(lags, ccf):
    fig, ax =plt.subplots(figsize=(9, 6))
    ax.plot(lags, ccf)
    ax.axhline(-2/np.sqrt(23), color='red', label='5 perc confidence interval')
    ax.axhline(2/np.sqrt(23), color='red')
    ax.axvline(x = 0, color = 'black', lw = 1)
    ax.axhline(y = 0, color = 'black', lw = 1)
    ax.axhline(y = np.max(ccf), color = 'blue', lw = 1, 
    linestyle='--', label = 'highest +/- correlation')
    ax.axhline(y = np.min(ccf), color = 'blue', lw = 1, 
    linestyle='--')
    ax.set(ylim = [-1, 1])
    ax.set_title('Cross Correlation of ' + SearchCategory + ' tweet vs nors case frequency', weight='bold', fontsize = 15)
    ax.set_ylabel('Correlation Coefficients', weight='bold', 
    fontsize = 12)
    ax.set_xlabel('Time Lags', weight='bold', fontsize = 12)
    plt.legend()
    plt.show()
    
ccf_plot(lags, ccf_caseFreq)