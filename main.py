from flask import  Flask,redirect,url_for,render_template,request,session
import os
import zipfile
import pandas as pd
import plotly.graph_objs as go
import json
import plotly
import plotly.graph_objs as go
import time
import schedule
from apscheduler.scheduler import Scheduler
import numpy as np


import atexit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from pymongo import MongoClient
import pickle


app = Flask(__name__)
cron = Scheduler(daemon=True)

cron.start()
df=pd.DataFrame()
df_grouped=pd.DataFrame()
df_new=pd.DataFrame()

os.environ['KAGGLE_USERNAME'] = "jagpreet796"
os.environ['KAGGLE_KEY'] = "d4f77b92179bfa82065bccfaa9c1dc54"
os.system("kaggle datasets download -d sudalairajkumar/novel-corona-virus-2019-dataset")
zf = zipfile.ZipFile('novel-corona-virus-2019-dataset.zip')
df = pd.read_csv(zf.open('covid_19_data.csv'))
client = MongoClient("mongodb+srv://m220-student:m220-mongodb-basics@mflix-sa05a.mongodb.net/DataProgrammingTestProject?retryWrites=true&w=majority")
collection=client["DataProgrammingTestProject"]
db = collection["NewCovidData2020"]
db.drop()
records_ = df.to_dict(orient = "records")
result = db.insert_many(records_)
x=db.count_documents({})
print("number of records",x)
headData = db.find()
row_list = []
for i in headData:
    row_list.append(i)


df_database=pd.DataFrame(row_list)

# df.drop("_id",axis=1,inplace=True)

df['ObservationDate']=pd.to_datetime(df['ObservationDate'])
df_grouped=df.groupby(['ObservationDate','Country/Region']).sum()


df_new=df_grouped.xs(df['ObservationDate'].max())

data= df.groupby(["ObservationDate"])['Confirmed','Deaths', 'Recovered'].sum().reset_index()
print(data.tail())
x_data=pd.DataFrame(data.index)
y_data=pd.DataFrame(data.Confirmed)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=10)
rmses = []
degrees = np.arange(1, 15)
min_rmse, min_deg = 1e10, 0
for deg in degrees:

    poly_features = PolynomialFeatures(degree=deg,include_bias=False)
    x_poly_train = poly_features.fit_transform(x_train)

    poly_reg = LinearRegression()
    poly_reg.fit(x_poly_train, y_train)

    x_poly_test = poly_features.fit_transform(x_test)
    poly_predict = poly_reg.predict(x_poly_test)
    poly_mse = mean_squared_error(y_test, poly_predict)
    poly_rmse = np.sqrt(poly_mse)
    rmses.append(poly_rmse)

    if min_rmse > poly_rmse:
        min_rmse = poly_rmse
        min_deg = deg

print('Best degree {} with RMSE {}'.format(min_deg, min_rmse))
poly = PolynomialFeatures(degree=min_deg)
x_transformed_data = poly.fit_transform(x_data)
poly_reg = LinearRegression()
poly_reg.fit(x_transformed_data, y_data)
poly_reg.predict((poly.fit_transform([[len(data) - 1]])))
filename = "finalized_model.pickle"
filename_2 = "poly.pickle"
pickle.dump(poly, open(filename_2, "wb"))
pickle.dump(poly_reg, open(filename, "wb"))
trial = len(data)
print(trial)
model = pickle.load(open("finalized_model.pickle", "rb"))
poly_loaded = pickle.load(open("poly.pickle", "rb"))
print("First time lets see", model.predict(poly_loaded.fit_transform([[trial-1]])))
print("lets see")
# pickle.dump(lm, open(filename, "wb"))
print(data.tail())

















@cron.interval_schedule(minutes=600)


def get_data():
    os.system("kaggle datasets download -d sudalairajkumar/novel-corona-virus-2019-dataset")
    zf = zipfile.ZipFile('novel-corona-virus-2019-dataset.zip')

    df_updated = pd.read_csv(zf.open('covid_19_data.csv'))
    client_new = MongoClient("mongodb+srv://m220-student:m220-mongodb-basics@mflix-sa05a.mongodb.net/DataProgrammingTestProject?retryWrites=true&w=majority")
    collection_new = client_new["DataProgrammingTestProject"]
    db_new = collection_new["NewCovidData2020"]
    db_new.drop()
    records_new_ = df_updated.to_dict(orient="records")
    result_new = db.insert_many(records_new_)
    x_new = db.count_documents({})
    print("number of records", x_new)
    headData_new = db.find()
    row_list_new = []
    for i in headData_new:
        row_list_new.append(i)

    df_updated_database=pd.DataFrame(row_list_new)

    df_updated_database.drop("_id", axis=1, inplace=True)

    df.update(df_updated)
    df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])
    df_grouped_updated = df.groupby(['ObservationDate', 'Country/Region']).sum()
    df_grouped.update(df_grouped_updated)

    df_new_updated = df_grouped.xs(df['ObservationDate'].max())
    df_new.update(df_new_updated)
    data = df.groupby(["ObservationDate"])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
    x_data = pd.DataFrame(data.index)
    y_data = pd.DataFrame(data.Confirmed)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=10)
    rmses = []
    degrees = np.arange(1, 15)
    min_rmse, min_deg = 1e10, 0

    for deg in degrees:

        poly_features = PolynomialFeatures(degree=deg, include_bias=False)
        x_poly_train = poly_features.fit_transform(x_train)

        poly_reg = LinearRegression()
        poly_reg.fit(x_poly_train, y_train)

        x_poly_test = poly_features.fit_transform(x_test)
        poly_predict = poly_reg.predict(x_poly_test)
        poly_mse = mean_squared_error(y_test, poly_predict)
        poly_rmse = np.sqrt(poly_mse)
        rmses.append(poly_rmse)

        if min_rmse > poly_rmse:
            min_rmse = poly_rmse
            min_deg = deg

    print('Best degree {} with RMSE {}'.format(min_deg, min_rmse))
    poly = PolynomialFeatures(degree=min_deg)
    x_data = poly.fit_transform(x_data)
    poly_reg = LinearRegression()
    poly_reg.fit(x_data, y_data)
    poly_reg.predict((poly.fit_transform([[len(data) - 1]])))
    filename = "finalized_model.pickle"
    filename_2 = "poly.pickle"
    pickle.dump(poly, open(filename_2, "wb"))
    pickle.dump(poly_reg, open(filename, "wb"))
    model = pickle.load(open("finalized_model.pickle", "rb"))
    poly_loaded = pickle.load(open("poly.pickle", "rb"))

    trial = len(data)
    print(trial)
    print("Lets get this party started with mongodb", model.predict(poly_loaded.fit_transform([[trial]])))
    filename = "finalized_model.pickle"
    # pickle.dump(lm, open(filename, "wb"))
    print(data.tail())

    print("file updated")
@app.route('/',methods=['POST','GET'])
def imp():
    if request.method =='POST':
        try:
            day=float(request.form['day'])
            filename = "finalized_model.pickle"
            load_model = pickle.load(open(filename, 'rb'))
            poly_loaded = pickle.load(open("poly.pickle", "rb"))
            prediction = load_model.predict(poly_loaded.fit_transform([[day]]))
            print("Inside the html",prediction[0][0])
            num=x_data.tail(5).values.tolist()
            cases=y_data.tail(5).values.tolist()
            print("Inside predictor",num[0][0])
            schedule.every(600).minutes.do(get_data)


            return render_template("result.html",prediction=round(prediction[0][0]),num_cases=zip(num,cases))
        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'


    else:

        schedule.every(600).minutes.do(get_data)
        return render_template("forecast.html")





@app.route('/map')
def first():


    # data = [
    #     go.Bar(x=df['Country/Region'],y=df['Confirmed'])
    #
    # ]
    df_new['text']='<br>'+'Confirmed :' + df_new['Confirmed'].astype(str) + '<br>'+'Recovered :' + df_new['Recovered'].astype(str) + '<br>' +'Deaths :' + df_new['Deaths'].astype(str)
    fig = go.Choropleth(
        locations=df_new.index,  # Spatial coordinates
        z=df_new['Confirmed'],  # Data to be color-coded
        locationmode='country names',  # set of locations match entries in `locations`
        colorscale='Blues',
        showlegend=False,

        text=df_new['text'],
        hovertext=df_new.index,
        hovertemplate="Country:%{hovertext},%{text}",
        colorbar_title='Confirmed<br>Cases',




    )
    d4 = [fig]
    graphJSON_1 = json.dumps(d4, cls=plotly.utils.PlotlyJSONEncoder)


    schedule.every(600).minutes.do(get_data)
    return render_template('index.html',
                           graphJSON=graphJSON_1)

@app.route('/scatter_1')
def second():
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(y=data['Confirmed'].values, x=data['ObservationDate'].values, name='Confirmed'))
    # fig.add_trace(go.Scatter(y=data['Deaths'].values, x=data['ObservationDate'].values, name='Deaths'))
    # fig.add_trace(go.Scatter(y=data['Recovered'].values, x=data['ObservationDate'].values, name='Recovered'))
    fig=go.Scatter(y=data['Confirmed'], x=data['ObservationDate'])
    d = [fig]
    graphJSON_1 = json.dumps(d, cls=plotly.utils.PlotlyJSONEncoder)

    # fig = go.Scatter(y=data['Recovered'], x=data['ObservationDate'])
    # d2 = [fig]
    # graphJSON_3 = json.dumps(d2, cls=plotly.utils.PlotlyJSONEncoder)

    schedule.every(600).minutes.do(get_data)
    return render_template('index_2.html',
                           graphJSON=graphJSON_1)

@app.route('/scatter_2')
def third():


    fig = go.Scatter(y=data['Recovered'], x=data['ObservationDate'])
    d1 = [fig]
    graphJSON_2 = json.dumps(d1, cls=plotly.utils.PlotlyJSONEncoder)

    schedule.every(600).minutes.do(get_data)
    return render_template('index_3.html',
                           graphJSON=graphJSON_2)
@app.route('/scatter_3')
def fourth():
    fig = go.Scatter(y=data['Deaths'], x=data['ObservationDate'])
    d1 = [fig]
    graphJSON_2 = json.dumps(d1, cls=plotly.utils.PlotlyJSONEncoder)
    schedule.every(600).minutes.do(get_data)
    return render_template('index_4.html',
                           graphJSON=graphJSON_2)
















if __name__=='__main__':
    app.run(debug=True)



