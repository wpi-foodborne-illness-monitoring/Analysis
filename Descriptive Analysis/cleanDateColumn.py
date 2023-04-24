import pandas
import torch

# load data
print('loading data...')
NORS = torch.load('NORSWithCategories.pt')
TWEETS = torch.load('TweetsWithCategories.pt')

NORS['DateFirstIll'] = pandas.to_datetime(NORS['DateFirstIll'])
TWEETS['created_at'] = pandas.to_datetime(TWEETS['created_at'])

# save df as csv and .pt file
print('saving to csv and pt...')
TWEETS.to_csv('TweetsWithCategories.csv', index=False)
torch.save(TWEETS,'TweetsWithCategories.pt')

NORS.to_csv('NORSWithCategories.csv', index=False)
torch.save(NORS,'NORSWithCategories.pt')