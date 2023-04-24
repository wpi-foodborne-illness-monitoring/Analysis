import pandas
import torch

print('reading csv...')
df = pandas.read_csv('filteredPositiveTweets.csv')

symptoms,locations,foods,others = [],[],[],[]

# iterate through all rows
for index, row in df.iterrows():
	# add the parsed list into the new column
	if len(row['symptoms']) > 2:
		symptoms.append(row['symptoms'][1:-1].split(','))
	else:
		symptoms.append([])
	if len(row['locations']) > 2:
		locations.append(row['locations'][1:-1].split(','))
	else:
		locations.append([])
	if len(row['foods']) > 2:
		foods.append(row['foods'][1:-1].split(','))
	else:
		foods.append([])
	if len(row['others']) > 2:
		others.append(row['others'][1:-1].split(','))
	else:
		others.append([])

# drop the unclean columns
df.drop(columns=['symptoms','locations','foods','others'])

# add the cleaned columns
df['symptoms'] = symptoms
df['locations'] = locations
df['foods'] = foods
df['others'] = others

# save df
print('saving to csv and pt...')
df.to_csv('cleanedFilteredPositiveTweets.csv', index=False)
torch.save(df,'cleanedFilteredPositiveTweets.pt')