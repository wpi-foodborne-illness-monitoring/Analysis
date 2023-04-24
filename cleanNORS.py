import torch
import pandas
import regex
import re

print('reading file...')
df = pandas.read_excel('NORS.xlsx')

# regex the columns we need
stateDF = df.filter(regex=('^State.*'))
ingredientDF = df.filter(regex=('^ContaminatedIngredientName.*'))
foodNameDF = df.filter(regex=('^FoodName.*'))
ifsaCLevelDF = df.filter(regex=('^IFSACLevel.*'))
taxonomyConfirmation = df.filter(regex=('^Confirmed.*'))

def condense(bigDF):
	newColumn = []
	for index, row in bigDF.iterrows():
		# remove nan
		cleanedList = [x for x in list(row) if str(x) != 'nan']

		# seperate words, remove non-alphanumeric character, and lowercase
		cleanedList2 = []
		# iterate through all words
		for bigWord in cleanedList:
			# split by spaces and clean
			bigWord = re.sub(r'[^a-zA-Z0-9]', ' ', bigWord)
			for word in bigWord.split():
			    cleanedList2.append(word.lower())
		newColumn.append(cleanedList2)
	return newColumn

def simplifyTax(conList):
	if len(conList) == 0:
		return ''
	elif 'confirmed' in conList:
		return 'confirmed'
	else:
		return 'suspected'

filteredDF = df[['cdcid', 'DateFirstIll', 'EstimatedPrimary', 'DeathsInfo', 'HospitalInfo', 'CAFC_1']]

filteredDF['States'] = condense(stateDF)
filteredDF['Ingredients'] = condense(ingredientDF)
filteredDF['FoodNames'] = condense(foodNameDF)
filteredDF['IFSACLevel'] = condense(ifsaCLevelDF)
filteredDF['TaxonomyConfirmation'] = list(map(simplifyTax,condense(taxonomyConfirmation)))

print('saving to csv and pt...')
filteredDF.to_csv('cleanedNORSData.csv', index=False)
torch.save(filteredDF,'cleanedNORSData.pt')
