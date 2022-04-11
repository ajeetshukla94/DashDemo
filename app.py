import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dct
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from datetime import datetime as dt
import plotly.express as px
import nltk
import os

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)
server = app.server

complaint_df           = pd.read_csv(os.path.join("static/inputData/","complaints_processed.csv"))
unq_Gender             = ['Male','Female']
complaint_df['Gender'] = np.random.choice(unq_Gender,size=len(complaint_df),p=[0.35,0.65])
unq_product            = complaint_df['product'].unique().tolist()
unq_Gender             = complaint_df['Gender'].unique().tolist()
colums_list            = {'Gender':'Gender','product':'product'}

drop_down_css    = {'background':'white','text':'white','font-size': '18px'}
label_css        = {'background':'white','text':'#0066ff','text-align': 'center','font-weight': 'bold','font-size': '15px'}
table_cell_css   = {'padding': '5px', 'text-align': 'center',
                  'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                    'overflow': 'hidden', 'textOverflow': 'ellipsis'} 
table_header_css = {'padding-top': '12px', 'padding-bottom': '12px',
                    'text-align': 'center', 'background-color': 'white', 
                    'color': 'black', 'font-size': '15px', 
                    'font-weight': 'bold', 'text-transform': 'uppercase'}
table_data_css   = {'color': 'black', 'backgroundColor': 'white',
                  'font-size': '12px', 'max-height': '50px'}
table_label_css  = {'font-weight': "bold","text-align": "center", 'font-size': '25px'}
table_scroll_css = {'overflow-y': 'scroll', 'overflow-x': 'scroll', 'flex-wrap': 'wrap'}

def get_word_frequency(list_val,number_of_keyword=10):    
    allWords    = nltk.tokenize.word_tokenize(list_val)
    allWordDist = nltk.FreqDist(w.lower() for w in allWords)
    stopwords   = nltk.corpus.stopwords.words('english')
    allWordExceptStopDist = nltk.FreqDist(w.lower() for w in allWords if w not in stopwords)

    df_fdist = pd.DataFrame.from_dict(allWordExceptStopDist, orient='index')
    df_fdist.columns = ['Frequency']
    df_fdist.index.name = 'Keyword'
    df_fdist = df_fdist.reset_index()
    df_fdist.sort_values('Frequency',ascending=False,inplace=True)
    df_fdist=df_fdist.iloc[:number_of_keyword]
    bar_chart = px.bar(df_fdist,x=df_fdist['Keyword'],color=df_fdist['Keyword'],y='Frequency',) 
    return bar_chart
    
def get_dct_table(datatable):
    dct_table=html.Div([dct.DataTable(data=datatable.to_dict("rows"),
                    columns=[{"id": x, "name": x} for x in datatable.columns],
                    filter_action="native",
                    sort_action="native",
                    sort_mode= 'multi',
                    style_as_list_view=True,
                    page_size=10,
                    style_cell=table_cell_css,
                    style_header=table_header_css,
                    style_data=table_data_css)])
    return dct_table
    
def get_distribution_table(column_name, distribution_raw_data):
    c = distribution_raw_data[column_name].value_counts(dropna=False)
    p = distribution_raw_data[column_name].value_counts(dropna=False, normalize=True) 
    distribution_raw_data = pd.concat([c,p], axis=1, keys=['Count','%']) 
    distribution_raw_data = distribution_raw_data.reset_index()
    distribution_raw_data.columns = [column_name, 'Count', 'Percentage'] 
    distribution_raw_data.Percentage =distribution_raw_data.Percentage*100
    distribution_raw_data.sort_values ("Percentage", ascending=False)
    distribution_raw_data.Percentage=distribution_raw_data.Percentage.round(2)
    distribution_raw_data.Percentage=distribution_raw_data.Percentage.astype (str) + '%'
    distribution_raw_data.Count=distribution_raw_data.Count.apply(lambda x: "{0:,}".format(x) )    
    distribution_raw_data=get_dct_table(distribution_raw_data)
    return distribution_raw_data
    

    

    
    
df_copy = complaint_df.iloc[0:50]

app.layout = html.Div(style={'backgroundColor':'white'},children=[ 
    
    
    html.Div(children=[
        html.H2(children='Complaint Dashboard')
    ], style={'textAlign': 'center'}),
    
    
    ################### Filter box ###################### 
    html.Div(children=[
        
        html.Div(children=[
    
        
        html.Br(),
            
        html.Div(children=[                             
                html.Div(children=[
                    html.Label('Total Conversastion', style={'paddingTop': '.3rem','text-align': 'center','font-size': '20px'}),
                    html.H4(id='id1', style={'fontWeight': 'bold','text-align': 'center'}),
                ], className="twelve columns number-stat-box"),                      
        ],style={'margin-bottom':'1rem','display':'flex','width':"100%",
                 'flex-wrap':'wrap','justify-content':'space-between',
                'boxShadow': '#e3e3e3 4px 4px 2px', 
                 'border-radius': '10px', 'marginTop': '2rem'}),

            
        html.Label('Filter by Product',style=label_css),       
        dcc.Dropdown(id='product_type',options=[{'label': i, 'value': i}for i in unq_product],
                     multi=True,placeholder="Filter by product :",style=drop_down_css),
        html.Br(),
        html.Label('Filter by Gender',style=label_css),
        dcc.Dropdown(id='gender',options=[{'label': i, 'value': i}for i in unq_Gender],
                     multi=True,placeholder="Filter by gender :",style=drop_down_css),
        html.Br(),        

       
    ], className="three columns",style={'padding':'2rem', 'margin':'1rem', 
                                        'boxShadow': '#e3e3e3 4px 4px 2px', 
                                        'border-radius': '10px', 'marginTop': '2rem'} ),
        
        
        
        
    html.Div(children=[
        
        
#         html.Div(children=[           
                          
#                 html.Div(children=[
#                     html.H3(id='id1', style={'fontWeight': 'bold'}),
#                     html.Label('Total Conversastion', style={'paddingTop': '.3rem'}),
#                 ], className="six columns number-stat-box"),  
        
            
#         ],style={'margin-bottom':'1rem','display':'flex','width':"100%",
#                  'flex-wrap':'wrap','justify-content':'space-between'}),
        
        
        html.Div(children=[
            
            dcc.Tabs([
                
                dcc.Tab(label="Sample Data",children=[
                   html.Div(children=[html.Div(id="samleData"),
                                     ],style=table_scroll_css, className="twleve columns"),
                   
                ]),
                
                
                    
                dcc.Tab(label="Data Distribtuion",children=[               
                    
                  html.Div(children=[
                      html.Br(),
                      html.Label('Select Column for Distrbution',style=label_css),
                      dcc.Dropdown(id='selected_column',options=[{'label': i, 'value': i}for i in colums_list],
                      multi=False,placeholder="Select Column :",value='Gender',style=drop_down_css),
                      html.Br(),
                      html.Div(id="distributionData"),
                      
                      html.Br(),
                      dcc.Graph(id="barchart"),
                                     ],style=table_scroll_css, className="twleve columns"),
                ]),
                
                
                dcc.Tab(label="Narrative Overview",children=[               
                    
                  html.Div(children=[
                      
                      html.Br(),
                      html.Label('Keyword Distrbution',style=label_css),
                      dcc.Graph(id="wordbarchart"),
                                     ],style=table_scroll_css, className="twleve columns"),
                ]),
                
                
            ]),
            
            
        ],className="twelve columns")
    ],style={'display':'flex','flex-wrap':'wrap'}),
            
            

  
])])


@app.callback(
    dash.dependencies.Output('id1','children'),    
    dash.dependencies.Output('samleData','children'),    
    dash.dependencies.Output('distributionData','children'), 
    dash.dependencies.Output('barchart','figure'), 
    dash.dependencies.Output('wordbarchart','figure'),     
    [dash.dependencies.Input('product_type','value')],
    [dash.dependencies.Input('gender','value')],
    [dash.dependencies.Input('selected_column','value')]
)

def update_layout(product_type_val,gender_val,distribution_column):
    
    df_copy = complaint_df.copy()    
    if product_type_val is not None:
        if len(product_type_val)>0:
            df_copy = df_copy[df_copy['product'].isin(product_type_val)]        
    if gender_val is not None:
        if len(gender_val)>0:
            df_copy = df_copy[df_copy['Gender'].isin(gender_val)]
        
    
    Data_1 = get_distribution_table(distribution_column,df_copy)
    trn_df = df_copy.groupby([distribution_column]).size().reset_index(name='counts') 
    count_sum=trn_df.counts.sum()
    trn_df['Percentage'] =trn_df.counts.apply(lambda x:  100*x/float(count_sum)).values
    bar_chart = px.bar(trn_df,x=trn_df[distribution_column],
                       color=trn_df[distribution_column],y='counts',
                       text=trn_df['Percentage'].apply(lambda x: "{0:,}".format(x))) 
    
    id1=df_copy.shape[0]
    
    list_val=""
    for val in df_copy.iloc[0:50].narrative:
        list_val =list_val+" "+str(val)
    wordbarchart = get_word_frequency(list_val)  
    
    Data_2 = get_dct_table(df_copy.iloc[0:50])              
    return id1, Data_2,Data_1,bar_chart,wordbarchart
    
    
    
    
    
    
if __name__ == '__main__':
    app.run_server(debug=True)

