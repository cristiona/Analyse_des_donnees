#!/usr/bin/env python
# coding: utf-8

# # Prédiction du nombre d'antennes susceptibles d'etre installées dans l'avenir (prédictions sur 1 ans, en général et pour les principaux opérateurs (SFR, Orange, Bouygues, Free)

# In[1040]:


#Installation des libraries et modules nécessaires pour les prédictions
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install keras')
get_ipython().system('pip install psycopg2')
import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[1041]:


#Connexion à la base de données cible: PostgreSQL

try:
    #création d'une connexion à la BD postegre en 
    #spécifiant les paramètres de connexion
    conn = psycopg2.connect(
        host="localhost",
        database="observatoireFrance",
        user="user",
        password="monpass")

    #gestion des erreurs de connexion
except psycopg2.Error as e:
    #si la connexion échoue, afficher l'erreur
    print("Error connecting to the database:")
    print(e)
    #sinon, afficher la ligne suivante
else:
    print("Connection established successfully")


# #### Extraction des données de la base de données dans un dataframe (librairie Pandas)

# In[1042]:


#Extraction des données de la BD postegres dans un dataframe en utilisation la librairie Pandas et la connexion créee
df = pd.read_sql_query('select * from public."observatoireCible5G"',con=conn)


# In[1043]:


# df_sfr = pd.read_sql_query('SELECT * FROM public."observatoireCible5G" where operateur = \'SFR\'',con=conn)
# df_orange = pd.read_sql_query('SELECT * FROM public."observatoireCible5G" where operateur = \'ORANGE\'',con=conn)
# df_bouygues = pd.read_sql_query('SELECT * FROM public."observatoireCible5G" where operateur = \'BOUYGUES TELECOM\'',con=conn)


# In[1044]:


#visualisation des données extraites
df


# Nous comptons bien 90789 enregistrements et 19 attributs conformement au contenu de la table observatoireCible de la base de données. L'extraction des données dans le dataframe, df, a réussie.

# In[1045]:


#Vérification de quelques informations statistiques relatives aux attributs dont le type est "numérique"
#Le but est de répérer des outliers (des données anormales )
df.describe()


# In[1046]:


#Vérification de quelques informations relatives au type des attributs ainsi la présence ou non des données manquantes
#Notons bien qu'il s'agit d'une vérification. La qualité des données a été traitée avec Talend en faisant de l'ETL
#https://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm
df.info()


# #### Feature engineering

# In[1047]:


#Suppression des attributs non nécéssaires à la tache de prédiction
df = df.drop(['sup_id', 'id', 'nom_deprtmt','nom_region','date_maj','sta_nm_anfr','nat_id','tpo_id', 'com_cd_insee', 'coordonnees','coord','statut'], axis=1)


# In[1048]:


#Traitement des données relatives aux dates: transformer la date au format Date de la librarie Pandas 
#pour faciliter l'utilisation des données pour les étapes de prédiction et de visualisation
df['emr_dt'] = pd.to_datetime(df['emr_dt'])
df.dtypes


# In[1049]:


def df_split(df, nom_operateur):
    df_x = df[df['operateur'] == nom_operateur]
    return df_x


# In[1051]:


#Creation de dataframes en fonction des differents operateurs de téléphonies mobiles
df_sfr = df_split(df, 'SFR')
df_orange = df_split(df, 'ORANGE')
df_bouygues = df_split(df, 'BOUYGUES TELECOM')
df_free = df_split(df, 'FREE MOBILE')


# In[1054]:


#Traitement des données relatives aux dates: création d'un nouvel attribut stockant 
# le nombre d'antennes installées par jour ==> utile pour les prédictions

def counting_nbr_install_per_day(df):
    #compter le nombre d'enregistrements correspondant aux dates distinctes ==> comptage du nombre d'antennes installées par jour
    df_x = df['emr_dt'].value_counts().to_frame(name='count').reset_index().sort_values(by=['emr_dt'])
    #pour le nouveau dataframe, chnager l'index qui s'auto-incrémente en index de format date
    df_x.set_index(df_x['emr_dt'], inplace=True)
    #changer la fréquence des dates en journalier pour faciliter les prédictions
    df_x.asfreq('D')
    #etant donné que les données sont journalières, de nouveaux enregistrements se sont créees car les données ne sont pas 
    #toutes journalières, pour les nouvaux enregistrements, le count est NaN
    #il est donc important de remplacer les données manquantes par 0 pour faciliter la prédiction
    df_x['count'] = df_x['count'].fillna(0)
    #suppression de l'attribut date car elle est désormais l'index
    df_x = df_x.drop('emr_dt', axis=1)
    return df_x


# In[1055]:


#Application de la fonction du count sur les dataframe contenant les données de chacun des opérateurs
count_days = counting_nbr_install_per_day(df)
count_days_sfr = counting_nbr_install_per_day(df_sfr)
count_days_orange =counting_nbr_install_per_day(df_orange)
count_days_bouygues =counting_nbr_install_per_day(df_bouygues)
count_days_free =counting_nbr_install_per_day(df_free)


# #### Visualisation du nombre d'installation par jour et par opérateur

# In[1059]:


def plot_df(df, x, y):
    plt.figure(figsize=(15,4), dpi=100)
    plt.plot(x, y)
    plt.title("Nombre d'installation d'antennes en France et DROM d'avril 2021 à novembre 2023")
    plt.ylabel("Nombre d'antennes installées")
    plt.xlabel("Date")
    plt.legend()
    plt.show()
    
plot_df(count_days, x=count_days.index, y=count_days['count'])


# #### SFR 

# In[1060]:


plot_df(count_days_sfr, x=count_days_sfr.index, y=count_days_sfr['count'])


# #### Orange 

# In[1061]:


plot_df(count_days_orange, x=count_days_orange.index, y=count_days_orange['count'])


# #### Bouygues 

# In[1062]:


plot_df(count_days_bouygues, x=count_days_bouygues.index, y=count_days_bouygues['count'])


# #### Free

# In[1063]:


plot_df(count_days_free, x=count_days_free.index, y=count_days_free['count'])


# # Application des modèles de prédiction et de forecast

# ## Modèle Prophet

# In[1068]:


#Ajout de la date comme index à notre Dataframe
def add_index(df_x):
    df_x["dt"] = df_x.index


# In[1069]:


#Application de la fonction add_index à chacun des Dataframes
add_index(count_days)
add_index(count_days_sfr)
add_index(count_days_orange)
add_index(count_days_bouygues)
add_index(count_days_free)


# In[1070]:


# Le modèle Prophet impose une convention de nommage des attributs, index inclus
# création de nouveaux dataframes avec les noms d'attributs convenables (ds pour la data 
# et y pour les données)
count = pd.DataFrame()
count_sfr = pd.DataFrame()
count_orange = pd.DataFrame()
count_bouygues = pd.DataFrame()
count_free = pd.DataFrame()


# In[1071]:


#Fonction permettant de faire la copie des donées ds Dataframes originales
#vers les Dataframes répondant aux conventions de nommage
def copy_df_new_col(df_new, df_old):
    df_new[['ds', 'y']]=df_old[['dt', 'count']].copy()
    df_new.reset_index(drop=True)


# In[1072]:


copy_df_new_col(count,count_days)
copy_df_new_col(count_sfr,count_days_sfr)
copy_df_new_col(count_orange,count_days_orange)
copy_df_new_col(count_bouygues,count_days_bouygues)
copy_df_new_col(count_free,count_days_free)


# In[1097]:


#Fonction de paramétrage du modèle Prophet
from prophet import Prophet

#Cette fonction prend en entrée le dataframe
#la période (int), dans notre cas, le nombre de jours future pour lesquelles faire la prédiction
#size = le len du dataframe pour permettre de completer le datagrame exsitant avec les nouvelles prédictions
def prophet_fit_predict(df,periods, size):
    #Instanciation
    model = Prophet()

    #La fonction fit() du modèle 
    #permet de faire l'apprentissage sur les données mis en entrée
    model.fit(df)
    
    #création d'un ouveau Dataframe dans lequel est stocké des dates antérieurs
    #aux dates sur lesquelles ont été faites l'apprentissage
    future = model.make_future_dataframe(periods=periods)
    
    #Prédiction sur les données futures = forecast
    forecast = model.predict(future)
    
    #Arrondissement des données prédites (car type float)
    forecast_prophet = forecast[['ds','yhat']].iloc[size:,:].round()
    #Pour toute donnée prédite, inféreieure à 0, 
    #la remplacer par 0
    forecast_prophet.loc[forecast_prophet['yhat'] < 0, 'yhat'] = 0
    #Faciliter l'insertion dans la base de données en faisant du Dataframe
    #une liste de valeurs
    forecast_prophet_list = forecast_prophet.values.tolist()

    #Visualisation des données originales et les données prédites
    plt.plot(forecast_prophet['ds'],forecast_prophet['yhat'])
    plt.plot(count['y'])
    plt.show()


# #### Forecast général (tout opérateur confondu) 

# In[1093]:


#Afficher le nombre d'enregistrements du dataframe pour renseigner le 
#apramètre size
print(len(count))


# In[1094]:


prophet_fit_predict = prophet_fit_predict(count,365,804)
prophet_fit_predict_gene_list = forecast_prophet_list


# #### Forecast SFR 

# In[1095]:


print(len(count_sfr))


# In[1098]:


prophet_fit_predict_sfr = prophet_fit_predict(count_sfr,365,718)
prophet_fit_predict_sfr_list = forecast_prophet_list


# #### Forecast Bouygues 

# In[1099]:


print(len(count_bouygues))


# In[1100]:


prophet_fit_predict_bouygues = prophet_fit_predict(count_bouygues,365,522)
prophet_fit_predict_bouygues_list = forecast_prophet_list


# #### Forecast Free 

# In[1101]:


print(len(count_free))


# In[1102]:


prophet_fit_predict_free = prophet_fit_predict(count_free,365,298)
prophet_fit_predict_free_list = forecast_prophet_list


# #### Forecast Orange 

# In[1103]:


print(len(count_orange))


# In[1104]:


prophet_fit_predict_orange = prophet_fit_predict(count_orange,365,730)
prophet_fit_predict_orange_list = forecast_prophet_list


# Les prédictions avec le modèle Prophète semble, selon le bon sens, etre peu représentatives de la réalité. Selon la tendance générale, le nombre d'antennes devrait etre supérieur aux données prédites notamment à cause de l'essor de la 5G. 
# 
# La fiabilité et la performance des modèle de machine learning spécialisés dans les séries temporelles sont toujours en cours d'optimisation. 

# ## Modèle ARIMA

# In[1105]:


# Pour manipuler le modèle ARIMA, nous allons manipuler les données
#mensuelles

#Feature engineering sur les dates pour faire la somme
#des antennes installées par mois

#Création d'un nouvel attribut, contennt la date au fomrat MOIS-ANNEE
df['month_year'] = df['emr_dt'].dt.strftime('%m-%Y')
#Trier les données en ordrer croissant en se basant sur les mois/annee
df.sort_values(by='month_year', ascending=True, inplace=True)


# In[1123]:


#refaire le split en fonction des opérateurs

df_sfr = df_split(df, 'SFR')
df_orange = df_split(df, 'ORANGE')
df_bouygues = df_split(df, 'BOUYGUES TELECOM')
df_free = df_split(df, 'FREE MOBILE')


# In[1130]:


#Fonction pour compter le nombre d'antennes installées par mois

def count_install_per_month(df):
    #Compter le nombre d'installations par mois
    count = df['month_year'].value_counts().to_frame(name='count').reset_index().sort_values(by=['month_year'])
    #creation d'un nouvel attribut pour stocker les dates: pour les manipuler
    count['m_y'] = pd.to_datetime(count['month_year'])
    #extraction du mois
    count['month'] = count['m_y'].dt.strftime('%m')
    #extraction de l'annee ==> pour mieux ordonner et faciliter les traitements
    count['year'] =count['m_y'].dt.strftime('%Y')

    #ordonner le décompte par mois et années par ordre chronologique
    count = count.sort_values(by=[ 'year', 'month'])
    count.groupby('year')
    count = count.set_index("m_y")
    count = count.drop(['month', 'year', 'month_year'], axis=1)
    return(count)


# In[1131]:


#Application de la fonction sur les données mensuelles
count_month = count_install_per_month(df)
count_month_sfr = count_install_per_month(df_sfr)
count_month_orange = count_install_per_month(df_orange)
count_month_bouygues = count_install_per_month(df_bouygues)
count_month_free = count_install_per_month(df_free)


# In[1133]:


#Visualisation
def viz_per_month(df):
    plt.figure(figsize= (10,6))
    plt.plot(df)
    plt.xlabel('Month-Year')
    plt.ylabel('Nombre d"antennes installées')
    plt.title('Installation des antennes en France et DROM entre Avril 2021 et Novembre 2023')


# #### Forecast mensuel général 

# In[1134]:


viz_per_month(count_month)


# #### Forecast mensuel SFR 

# In[1136]:


viz_per_month(count_month_sfr)


# #### Forecast mensuel Orange 

# In[1137]:


viz_per_month(count_month_orange)


# #### Forecast mensuel Bouygues 

# In[1138]:


viz_per_month(count_month_bouygues)


# #### Forecast mensuel Free 

# In[1140]:


viz_per_month(count_month_free)


# Apprentissage avec le modèle ARIMA: test avec differents paramètres avec chaucn des dataframes

# In[1170]:


#Premier test avec un order de 2,1,2
from statsmodels.tsa.arima.model import ARIMA

def arima_order_212(df_list):
    
    models = []
    results_ARIMA = []
    
    for df in df_list:
        model = ARIMA(df, order=(2, 1, 2), freq='MS')
        results = model.fit()
        models.append(model)
        results_ARIMA.append(results)
    return models, results_ARIMA

    df_list = [df, df1, df2, df3, df4]
    models, results_ARIMA = arima_order_212(df_list)


# In[1186]:


df_list = count_month, count_month_sfr, count_month_orange, count_month_bouygues, count_month_free
models, results_ARIMA = arima_order_212(df_list)


# In[1189]:


#Visualisation
for i, (model, results) in enumerate(zip(models, results_ARIMA)):
    plt.figure(figsize=(10, 4))
    # Replace 'your_data_column_name' with the appropriate column name from your DataFrame
    plt.plot(df_list[i]['count'], label='Original Data')
    plt.plot(results.fittedvalues, color='red', label='ARIMA predictions')
    plt.title(f'ARIMA Model for DataFrame {i+1}')
    plt.legend()
    plt.show()


# In[1190]:


#Premier test avec un order de 2,1,2
from statsmodels.tsa.arima.model import ARIMA

def arima_order_210(df_list):
    
    models = []
    results_ARIMA = []
    
    for df in df_list:
        model = ARIMA(df, order=(2, 1, 0), freq='MS')
        results = model.fit()
        models.append(model)
        results_ARIMA.append(results)
    return models, results_ARIMA

    df_list = [df, df1, df2, df3, df4]
    models, results_ARIMA = arima_order_212(df_list)


# In[1191]:


models, results_ARIMA = arima_order_210(df_list)


# In[1192]:


#Visualisation
for i, (model, results) in enumerate(zip(models, results_ARIMA)):
    plt.figure(figsize=(10, 4))
    # Replace 'your_data_column_name' with the appropriate column name from your DataFrame
    plt.plot(df_list[i]['count'], label='Original Data')
    plt.plot(results.fittedvalues, color='red', label='ARIMA predictions')
    plt.title(f'ARIMA Model for DataFrame {i+1}')
    plt.legend()
    plt.show()


# In[1204]:


#Stocker dans des variables, les données prédites

result1, result2, result3, result4, result5 = results_ARIMA
result1 = results_ARIMA[0]
result2 = results_ARIMA[1]
result3 = results_ARIMA[2]
result4 = results_ARIMA[3]
result5 = results_ARIMA[4]


# In[1205]:


general = result1.forecast(steps=12)
sfr= result2.forecast(steps=12)
orange= result3.forecast(steps=12)
bouygues= result4.forecast(steps=12)
free= result5.forecast(steps=12)


# Le modèle ARIMA avec le paramètre Order 2,1,0 semble etre le plus optimale car l'apprentissage est le plus correct indépendameent des datasets sur lesquels sont faits les appprentissages

# In[1197]:


#Nous voulons visualier les données prédites et faire la comparaison 
#avec les données réelles

predictions_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)

print('Total no of predictions: ', len(predictions_diff))
predictions_diff.head()


# In[ ]:


predictions_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)

print('Total no of predictions: ', len(predictions_diff))
predictions_diff.head()


# In[1015]:


df_predictions =pd.DataFrame(predictions_diff, columns=['Predicted Values'])
pd.concat([count_days_sfr,df_predictions],axis =1).T


# In[1036]:


pred = results_ARIMA.predict(start = 30, end= 44) 


# In[1037]:


plt.plot(count_days_sfr)
plt.plot(pred)


# In[ ]:


# #Premier test avec un order de 2,1,2
# from statsmodels.tsa.arima.model import ARIMA

# def arima_order_212(df, df1, df2, df3, df4):
#     model = ARIMA(df, order=(2, 1, 2), freq = 'MS')
#     model1 = ARIMA(df1, order=(2, 1, 2), freq = 'MS')  
#     model2 = ARIMA(df2, order=(2, 1, 2), freq = 'MS')  
#     model3 = ARIMA(df3, order=(2, 1, 2), freq = 'MS')  
#     model4 = ARIMA(df4, order=(2, 1, 2), freq = 'MS')  

#     results_ARIMA = model.fit()  
#     results_ARIMA1 = model1.fit()  
#     results_ARIMA2 = model2.fit()  
#     results_ARIMA3 = model3.fit()  
#     results_ARIMA4 = model4.fit()  

# #     for i in 
#     plt.plot(df, label='Nombre réel d"antennes installées: GENERAL')
#     plt.plot(results_ARIMA.fittedvalues, label='Nombre prédit d"antennes installées: GENERAL')
    
#     plt.plot(df1, label='Nombre réel d"antennes installées: SFR')
#     plt.plot(results_ARIMA1.fittedvalues, label='Nombre prédit d"antennes installées: SFR')
    
#     plt.plot(df2, label='Nombre réel d"antennes installées: ORANGE')
#     plt.plot(results_ARIMA2.fittedvalues, label='Nombre prédit d"antennes installées: ORANGE')
    
#     plt.plot(df3, label='Nombre réel d"antennes installées: BOUYGUES')
#     plt.plot(results_ARIMA3.fittedvalues, label='Nombre prédit d"antennes installées: BOUYGUES')
    
#     plt.plot(df4, label='Nombre réel d"antennes installées: FREE')
#     plt.plot(results_ARIMA4.fittedvalues, label='Nombre prédit d"antennes installées: FREE')
            
#     plt.xlabel("Month-year")
#     plt.ylabel("Nombre d'antennes installées")
#     plt.legend()

#     plt.show()
# #     plt.plot(results_ARIMA.fittedvalues, color='red')


# In[1039]:


df_forecast =pd.DataFrame(pred, columns=['Forecast'])
df_forecast


# In[1007]:


predictions_diff_cumsum = predictions_diff.cumsum()
predictions_diff_cumsum.head()


# In[997]:


def stationarity_test(timeseries):
    # Get rolling statistics for window = 12 i.e. yearly statistics
    rolling_mean = timeseries.rolling(window = 12).mean()
    rolling_std = timeseries.rolling(window = 12).std()
    
    # Plot rolling statistic
    plt.figure(figsize= (10,6))
    plt.xlabel('Years')
    plt.ylabel('No of Air Passengers')    
    plt.title('Stationary Test: Rolling Mean and Standard Deviation')
    plt.plot(timeseries, color= 'blue', label= 'Original')
    plt.plot(rolling_mean, color= 'green', label= 'Rolling Mean')
    plt.plot(rolling_std, color= 'red', label= 'Rolling Std')   
    plt.legend()
    plt.show()
    
    # Dickey-Fuller test
    print('Results of Dickey-Fuller Test')
    df_test = adfuller(timeseries)
    df_output = pd.Series(df_test[0:4], index = ['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in df_test[4].items():
        df_output['Critical Value (%s)' %key] = value
    print(df_output)


# In[998]:


stationarity_test(count_days_sfr)


# In[999]:


df_diff = count_days_sfr.diff(periods = 1) # First order differencing
plt.xlabel('Years')
plt.ylabel('No of Air Passengers')    
plt.title('Convert Non Stationary Data to Stationary Data using Differencing ')
plt.plot(df_diff)


# In[866]:


plt.plot(count_days_sfr)


# In[890]:


count_days_sfr_month['month_year'] = count_days_sfr.index.dt.strftime('%m-%Y')
count_days_sfr_month.sort_values(by='month_year', ascending=True, inplace=True)


# In[880]:


import seaborn as sns
from statsmodels.tsa.stattools import adfuller,acf, pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from pylab import rcParams

rcParams['figure.figsize'] = 10, 6

df_temp = count_days_sfr.copy()

df_temp['Year'] = pd.DatetimeIndex(df_temp.index).year
df_temp['Month'] = pd.DatetimeIndex(df_temp.index).month
df_temp


# In[889]:


count_days_sfr['Month'] = count_days_sfr.index
count_days_sfr['Month'] = pd.to_datetime(count_days_sfr.Month)
count_days_sfr = count_days_sfr.set_index(count_days_sfr.Month)
count_days_sfr.drop('Month', axis = 1, inplace = True)
print('Column datatypes= \n',count_days_sfr.dtypes)
count_days_sfr


# In[881]:


# Stacked line plot
plt.figure(figsize=(10,10))
plt.title('Seasonality of the Time Series')
sns.pointplot(x='Month',y='count',hue='Year',data=df_temp)


# In[882]:


decomposition = sm.tsa.seasonal_decompose(count_days_sfr, model='additive') 
fig = decomposition.plot()


# In[845]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# In[960]:


import math

df = math.exp(df_log)


# In[848]:


df_sfr = count_days_sfr.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
df_sfr = scaler.fit_transform(df_sfr)
df_sfr


# In[849]:


# split into train and test sets
train_size = int(len(df_sfr) * 0.8)
test_size = len(df_sfr) - train_size
train, test = df_sfr[0:train_size,:], df_sfr[train_size:len(df_sfr),:]
print(len(train), len(test))


# In[851]:


# convert an array of values into a dataset matrix
def create_dataset(df, look_back=1):
	X, Y = [], []
	for i in range(len(df)-look_back-1):
		a = df[i:(i+look_back), 0]
		X.append(a)
		Y.append(df[i + look_back, 0])
	return np.array(X), np.array(Y)


# In[852]:


# reshape into X=t and Y=t+1
look_back = 1
X_train, Y_train = create_dataset(train, look_back)
X_test, Y_test = create_dataset(test, look_back)


# In[853]:


# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


# In[854]:


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2)


# In[856]:


# make predictions
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
Y_train = scaler.inverse_transform([Y_train])
testPredict = scaler.inverse_transform(testPredict)
Y_test = scaler.inverse_transform([Y_test])
# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(Y_train[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(Y_test[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# In[857]:


# shift train predictions for plotting
trainPredictPlot = np.empty_like(df_sfr)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df_sfr)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(df_sfr)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df_sfr))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# # Arima

# In[824]:


# from statsmodels.tsa.arima.model import ARIMA

# def model_arima_fit_predict(df, x, date_start, date_end):
#     model = ARIMA(df)
#     try:
#         model_fit = model.fit()
#         print(model_fit.summary())
#         if model_fit is None:
#             print("Fitted model not available, cannot make predictions")
#         fit_forecast = model_fit.forecast(x)
#         pred = fit_forecast(date_start,date_end, dynamic=True)
# #         print(pred)
#         return pred
#     except Exception as e:
#         print(f"Error with ARIMA model: {e}")
        


# In[825]:


# model_test_fit = model_arima_fit_predict(count_days_sfr['count'], 700,"2023-11-17", "2024-11-17")
# print(model_test_fit)


# In[826]:


# model_fit.summary()


# In[827]:


# def model_arima_predict(model_fit, x, date_start, date_end):
#     if model_fit is None:
#         print("Fitted model not available, cannot make predictions")
#         return None
#     pred = fit_forecast(date_start,date_end, dynamic=True)
#     return pred


# In[828]:


# model_test_predict = model_arima_predict(model_fit, count_days['count'],"2023-11-17", "2024-11-17")


# In[829]:


# model2 = ARIMA(count_days['count'])
# fit = model2.fit()

# # Forecast five steps from the end of `series`
# fit.forecast(700)

# # Forecast five steps starting after the tenth observation in `series`
# # Note that the `dynamic=True` argument specifies that it only uses the
# # actual data through the tenth observation to produce each of the
# # five forecasts
# pred = fit.predict("2023-11-17", "2024-11-17", dynamic=True)

# # # Add new observations (`new_obs`) to the end of the dataset
# # # *without refitting the parameters* and then forecast
# # # five steps from the end of the new observations
# # fit_newobs = fit.append(new_obs, refit=False)
# # fit_newobs.forecast(5)

# # # Apply the model and the fitted parameters to an
# # # entirely different dataset (`series2`) and then forecast
# # # five steps from the end of that new dataset
# # fit_newdata = fit.apply(series2)
# # fit_newdata.forecast(5)


# Visualisation

# In[806]:


# future_dates = pd.date_range(start='2023-11-17', periods=365, freq='D')

# try:
#     forecast = model_fit.get_forecast(steps=365)
#     forecast_values = forecast.predicted_mean
#     forecast_df = pd.DataFrame(forecast_values, index=future_dates)
# except Exception as e:
#     print(f"Error making predictions with the ARIMA model: {e}")
# # Print or use the forecasted data as needed
# print(forecast_df)


# In[807]:


# print(forecast)


# In[808]:


# plt.plot(count_days['count'], label='Original Data')
# plt.plot(forecast_df, label='Predictions')
# plt.legend()
# plt.show()


# In[55]:


#save to postgresql


# In[988]:


try:
    cur = conn.cursor()
#     cur.executemany("INSERT INTO public.predictions_prophet(dt, count_antennas) VALUES (%s, %s)", forecast_prophet_list)
    cur.executemany("INSERT INTO public.predictions_prophet_sfr(dt, count_antennas) VALUES (%s, %s)", prophet_fit_predict_sfr_list)
    cur.executemany("INSERT INTO public.predictions_prophet_orange(dt, count_antennas) VALUES (%s, %s)", prophet_fit_predict_orange_list)
    cur.executemany("INSERT INTO public.predictions_prophet_bouygues(dt, count_antennas) VALUES (%s, %s)", prophet_fit_predict_bouygues_list)
    cur.executemany("INSERT INTO public.predictions_prophet_free(dt, count_antennas) VALUES (%s, %s)", prophet_fit_predict_free_list)

#     cur.executemany("INSERT INTO public.predictions_prophet_(dt, count_antennas) VALUES (%s, %s)", forecast_prophet_list)

    conn.commit()
    print("Données enregistrées")
#     print(cur.rowcount(), " enregistrements insérés")
except(Exception, psycopg2.Error) as error:
    print("Erreur, les données n'ont pas été insérées: ", error)
finally:
    if conn:
        cur.close()
        conn.close()
        print("Connexion fermée")

