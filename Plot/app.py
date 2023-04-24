# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import torch
from operator import itemgetter
from datetime import date
import numpy as np

app = Dash(__name__)

colors = {
    'background': '#ea9999ff',
    'text': 'black'
}

tweetsDF = torch.load('TweetsWithCategories.pt')
NORSDF = torch.load('NORSWithCategories.pt')

def top30fig(NORSDF,tweetsDF,startDate,endDate):

    tweetsDF.index = pd.to_datetime(tweetsDF['created_at'])
    NORSDF.index = pd.to_datetime(NORSDF['DateFirstIll'])

    # filter the data to be a comparible timeframe
    tweetsDF = tweetsDF[tweetsDF.index.year > int(startDate[:4])]
    tweetsDF = tweetsDF[tweetsDF.index.year < int(endDate[:4])]
    tweetsDF = tweetsDF[tweetsDF.index.month > int(startDate[5:7])]
    tweetsDF = tweetsDF[tweetsDF.index.month < int(endDate[5:7])]
    tweetsDF = tweetsDF[tweetsDF.index.day > int(startDate[8:10])]
    tweetsDF = tweetsDF[tweetsDF.index.day < int(endDate[8:10])]


    NORSDF = NORSDF[NORSDF.index.year > int(startDate[:4])]
    NORSDF = NORSDF[NORSDF.index.year < int(endDate[:4])]
    NORSDF = NORSDF[NORSDF.index.month > int(startDate[5:7])]
    NORSDF = NORSDF[NORSDF.index.month < int(endDate[5:7])]
    NORSDF = NORSDF[NORSDF.index.day > int(startDate[8:10])]
    NORSDF = NORSDF[NORSDF.index.day < int(endDate[8:10])]

    tweetDic = {}
    for index, row in tweetsDF.iterrows():
        for token in row['foods']:
            if token in tweetDic.keys():
                tweetDic[token] += 1
            else:
                tweetDic[token] = 0

    resultDict = dict(sorted(tweetDic.items(), key = itemgetter(1), reverse = True)[:30])
    resultTweetDict = {'tokens':list(resultDict.keys()), 'counts':list(resultDict.values())}

    norsDic = {}
    for index, row in NORSDF.iterrows():
        for token in row['Ingredients']:
            if token in norsDic.keys():
                norsDic[token] += 1
            else:
                norsDic[token] = 0

    resultDict = dict(sorted(norsDic.items(), key = itemgetter(1), reverse = True)[:30])
    resultNORSDict = {'tokens':list(resultDict.keys()), 'counts':list(resultDict.values())}

    colorList = []
    for token in resultTweetDict['tokens']:
        if token in resultNORSDict['tokens']:
            colorList.append('Shared Token')
        else:
            colorList.append('Unique')
    resultTweetDict['Token Group'] = colorList

    colorList = []
    for token in resultNORSDict['tokens']:
        if token in resultTweetDict['tokens']:
            colorList.append('Shared Token')
        else:
            colorList.append('Unique')
    resultNORSDict['Token Group'] = colorList

    tweetFig = px.bar(resultTweetDict, x='tokens', y='counts',color="Token Group",color_discrete_map={'Shared Token': '#ea9999','Unique': '#a4c2f4'})
    
    norsFig = px.bar(resultNORSDict, x='tokens', y='counts',color="Token Group",color_discrete_map={'Shared Token': '#ea9999','Unique': '#a4c2f4'})
    norsFig.update_layout(showlegend=False)
    return tweetFig, norsFig

def foodCategoryFig(NORS,TWEETS,startDate,endDate,SearchCategory,isIWP):
    # filter by food
    NORS = NORS[pd.DataFrame(NORS.Category.tolist()).isin([SearchCategory]).any(1).values]
    TWEETS = TWEETS[pd.DataFrame(TWEETS.Category.tolist()).isin([SearchCategory]).any(1).values]

    TWEETS.index = pd.to_datetime(TWEETS['created_at'])
    NORS.index = pd.to_datetime(NORS['DateFirstIll'])

    # filter the data to be a comparible timeframe
    TWEETS = TWEETS[TWEETS.index.year > int(startDate[:4])]
    TWEETS = TWEETS[TWEETS.index.year < int(endDate[:4])]
    TWEETS = TWEETS[TWEETS.index.month > int(startDate[5:7])]
    TWEETS = TWEETS[TWEETS.index.month < int(endDate[5:7])]
    TWEETS = TWEETS[TWEETS.index.day > int(startDate[8:10])]
    TWEETS = TWEETS[TWEETS.index.day < int(endDate[8:10])]


    NORS = NORS[NORS.index.year > int(startDate[:4])]
    NORS = NORS[NORS.index.year < int(endDate[:4])]
    NORS = NORS[NORS.index.month > int(startDate[5:7])]
    NORS = NORS[NORS.index.month < int(endDate[5:7])]
    NORS = NORS[NORS.index.day > int(startDate[8:10])]
    NORS = NORS[NORS.index.day < int(endDate[8:10])]

    # filter out iwp
    if isIWP == 'false':
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

    X_axis = []

    for tup in listAllDates:
        X_axis.append(str(tup[0])+'-'+str(tup[1]))

    dataset = []
    for i in range(len(NORS_perc)):
        dataset.append('NORS')
    for i in range(len(TWEETS_perc)):
        dataset.append('Tweets')

    resultDict = {'percent':list(NORS_perc)+list(TWEETS_perc),'date':list(X_axis)+list(X_axis), 'Dataset':dataset}

    fig = px.bar(resultDict, x='date', y='percent',color="Dataset",color_discrete_map={'NORS': '#ea9999','Tweets': '#a4c2f4'},barmode='group')

    return fig

app.layout = html.Div(style={'backgroundColor': colors['background']},children=[
    html.H1(
        children='USDA Foodborne Illness Outbreak Detection and Visualization',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(style={'display': 'flex'}, children=[
        html.Div(children='Advisors: Elke Rundensteiner (DS/CS), Oren Mangoubi (MA)', style={
            'textAlign': 'center',
            'flex': '33%',
            'color': colors['text']
        }),
        html.Div(children='MQP Students: Katy Hartmann (DS), Timothy Kwan (CS/DS), Jasmine Laber (DS), Anne Lapsley (DS/MA), Liam Rathke (CS)', style={
            'textAlign': 'center',
            'flex': '33%',
            'color': colors['text']
        }),
        html.Div(children='PhD Students: Ruofan Hu, Dongyu Zhang', style={
            'textAlign': 'center',
            'flex': '33%',
            'color': colors['text']
        })
    ]),

    html.Div(style={'display': 'flex','backgroundColor': 'white'}, children=[
        html.Div(style={'flex':'100%','textAlign': 'center'},children=[
            html.H3(
                children='Date Range:',
                style={
                    'textAlign': 'center'
                }
            ),
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date=date(2016, 1, 1),
                end_date=date(2023, 12, 31)
                )
            ]),
        html.Div(style={'flex':'0%'},children=[])
    ]),

    html.Br(),

    html.Div(style={'display': 'flex','backgroundColor': 'white'}, children=[
        html.H2(children='Token Analysis',style={'textAlign': 'center','backgroundColor': '#a4c2f4'}),
        html.Div(style={'flex':'47%'},children=[
            html.H2(
                children='Top 30 NORS Tokens',
                style={
                    'textAlign': 'center'
                }
            ),
            dcc.Graph(
                id='top-nors',
                figure=top30fig(NORSDF,tweetsDF,'2016-01-01','2023-12-31')[1]
            )]),
        html.Div(style={'flex':'53%'},children=[
            html.H2(
                children='Top 30 Tweet Tokens',
                style={
                    'textAlign': 'center'
                }
            ),
            dcc.Graph(
                id='top-tweets',
                figure=top30fig(NORSDF,tweetsDF,'2016-01-01','2023-12-31')[0]
            )])
    ]),

    html.Br(),

    html.Div(style={'display': 'flex','backgroundColor': 'white'}, children=[
        html.H2(children='Food Category Analysis',style={'textAlign': 'center','backgroundColor': '#a4c2f4'}),
         html.Div(style={'flex':'100%'},children=[
            html.Div(style={'flex':'33%'},children=[
                html.H3(children='Food Category: '),
                dcc.Dropdown(
                    id='cat-dropdown',
                    style={'width': '33%'},
                    options=[
                    {'label':'fish','value':'fish'},
                    {'label':'shellfish','value':'shellfish'},
                    {'label':'dairy','value':'dairy'},
                    {'label':'egg','value':'egg'},
                    {'label':'meat','value':'meat'},
                    {'label':'poultry','value':'poultry'},
                    {'label':'fruit/nut','value':'fruit/nut'},
                    {'label':'vegetable','value':'vegetable'},
                    {'label':'oil/sugar','value':'oil/sugar'},
                    {'label':'grain/bean','value':'grain/bean'}],
                    placeholder='select category...'
                    ),
                html.H3(children='Filter: '),
                dcc.RadioItems(
                    id='cat-radio',
                    style={'width': '33%'},
                    options=[
                    {'label':'With I was Poisioned','value':'true'},
                    {'label':'Without I was Poisioned','value':'false'}])
                ]),
            html.H2(
                id='cat-heading',
                children='Percent poultry overtime',
                style={
                    'textAlign': 'center'
                }
            ),
            dcc.Graph(
                id='cat-overtime',
                figure=foodCategoryFig(NORSDF,tweetsDF,'2016-01-01','2023-12-31','poultry','true')
            )])

    ])
])

@app.callback(
    Output('top-nors','figure'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'))
def updateTopNORS(start_date,end_date):
    return top30fig(NORSDF,tweetsDF,start_date,end_date)[1]

@app.callback(
    Output('top-tweets','figure'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'))
def updateTopTweets(start_date,end_date):
    return top30fig(NORSDF,tweetsDF,start_date,end_date)[0]

@app.callback(
    Output('cat-overtime','figure'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'),
    Input('cat-dropdown','value'),
    Input('cat-radio','value'))
def updateCatOvertime(start_date,end_date,SearchCategory,isIWP):
    if SearchCategory == None:
        return foodCategoryFig(NORSDF,tweetsDF,start_date,end_date,'poultry',isIWP)
    return foodCategoryFig(NORSDF,tweetsDF,start_date,end_date,SearchCategory,isIWP)

@app.callback(
    Output('cat-heading','children'),
    Input('cat-dropdown','value'))
def updateCatHeading(category):
    if category == None:
        return 'Percent poultry overtime'
    return 'Percent ' + category + ' overtime'

if __name__ == '__main__':
    app.run_server(debug=True)
