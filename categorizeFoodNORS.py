import torch
import pandas
import json

# define the 
print('loading df...')
df = torch.load('cleanedNORSData.pt')

# import ingredients to categories dictionary
with open('ingredients.json', 'r') as fp:
    foodDict = json.load(fp)

# return word with and without s
def getSWords(word):
	# word has s in it already
	if word[-1] == 's':
		return word,word[:-1]
	# doesn't have an s in it
	else:
		return word,word+'s'

# return category of the food if found, othewise return -1
def getCategory(food):
	words = getSWords(food)
	if words[0] in foodDict:
		return foodDict[words[0]]
	elif words[1] in foodDict:
		return foodDict[words[1]]
	# not in the dictionary
	else: 
		return -1

# ----------------------------------start transformation------------------------------------------
newCategoryCol = []

# iterate through all rows
for index, row in df.iterrows():
	# initialize empty set to avoid replication
	categorySet = set()
	# iterate through ingredients
	for food in row['Ingredients']:
		category = getCategory(food)
		if category != -1:
			categorySet.add(category)
	# iterate through foods
	for food in row['FoodNames']:
		category = getCategory(food)
		if category != -1:
			categorySet.add(category)

	# add the categories to the new column list
	newCategoryCol.append(list(categorySet))

df['Category'] = newCategoryCol

# save df as csv and .pt file
print('saving to csv and pt...')
df.to_csv('NORSWithCategories.csv', index=False)
torch.save(df,'NORSWithCategories.pt')