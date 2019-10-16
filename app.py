# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("revisedstudentdata.csv")
df=df.fillna(69)
X = df[df.columns.difference(['Paper7', 'Semster_Name', 'Student_ID','Unnamed: 0'])]
Y=df['Paper7']

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

regressor = LinearRegression()  
regressor.fit(X_train, Y_train) 



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server=app.server

app.layout = html.Div([
        
    html.H1('Grade Predictor'),
        
    html.Div([   
    html.Label('Paper 1'),
    dcc.Slider(id='paper1-slider',
            min=0, max=100, step=1, value=50,
               marks={
        0: {'label': '0'},
            25: {'label': '25'},
        50: {'label': '50'},
        75: {'label': '75'},
        100: {'label': '100'},
                                     
    }),

html.Br(),
html.Label('Paper 2'),
dcc.Slider(id='paper2-slider',
            min=0, max=100, step=1, value=50,
               marks={
        0: {'label': '0'},
        25: {'label': '25'},
        50: {'label': '50'},
        75: {'label': '75'},
        100: {'label': '100'},
                                 
    }),

html.Br(),
html.Label('Paper 3'),
dcc.Slider(id='paper3-slider',
            min=0, max=100, step=1, value=50,
               marks={
        0: {'label': '0'},
        25: {'label': '25'},
        50: {'label': '50'},
        75: {'label': '75'},
        100: {'label': '100'},
                                
    }),

html.Br(),
html.Label('Paper 4'),
dcc.Slider(id='paper4-slider',
            min=0, max=100, step=1, value=50,
               marks={
        0: {'label': '0'},
        25: {'label': '25'},
        50: {'label': '50'},
        75: {'label': '75'},
        100: {'label': '100'},
                                
    }),

html.Br(),
html.Label('Paper 5'),
dcc.Slider(id='paper5-slider',
            min=0, max=100, step=1, value=50,
               marks={
        0: {'label': '0'},
        25: {'label': '25'},
        50: {'label': '50'},
        75: {'label': '75'},
        100: {'label': '100'},
                                
    }),

html.Br(),
html.Label('Paper 6'),
dcc.Slider(id='paper6-slider',
            min=0, max=100, step=1, value=50,
               marks={
        0: {'label': '0'},
        25: {'label': '25'},
        50: {'label': '50'},
        75: {'label': '75'},
        100: {'label': '100'},
                                
    }),
            
],className="pretty_container four columns"),

  html.Div([ 

    daq.Gauge(
        id='my-gauge',
        showCurrentValue=True,
        color={"gradient":True,"ranges":{"red":[0,30],"yellow":[30,60],"green":[60,100]}},
        label="Test Score",
        max=100,
        min=0,
        value=1
    ),
])
    ])


@app.callback(
    Output('my-gauge', 'value'),
    [Input('Paper1-slider', 'value'),
     Input('Paper2-slider', 'value'),
     Input('Paper3-slider', 'value'),
     Input('Paper4-slider', 'value'),
     Input('Paper5-slider', 'value'),
     Input('Paper6-slider', 'value')
     ])
def update_output_div(Paper1,
                      Paper2,
                      Paper3,
                      Paper4,
                      Paper5,
                      Paper6):
   X_case =pd.DataFrame({'Paper 1':[Paper1],'Paper 2':[Paper2],'Paper 3':[Paper3],'Paper 4':[Paper4],'Paper 5':[Paper5],'Paper 6':[Paper6]})
   Y_case = regressor.predict(X_case)

   return Y_case[0]


if __name__ == '__main__':
    app.run_server()
