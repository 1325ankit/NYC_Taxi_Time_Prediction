#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement
# At some point or the other almost each one of us has used an Ola or Uber for taking a ride. 
# 
# Ride hailing services are services that use online-enabled platforms to connect between passengers and local drivers using their personal vehicles. In most cases they are a comfortable method for door-to-door transport. Usually they are cheaper than using licensed taxicabs. Examples of ride hailing services include Uber and Lyft.
# 
# 
# To improve the efficiency of taxi dispatching systems for such services, it is important to be able to predict how long a driver will have his taxi occupied. If a dispatcher knew approximately when a taxi driver would be ending their current ride, they would be better able to identify which driver to assign to each pickup request.
# 
# In this competition, we are challenged to build a model that predicts the total ride duration of taxi trips in New York City.

# ## 1. Exploratory Data Analysis
# Let's check the data files! According to the data description we should find the following columns:
# 
#  - **id** - a unique identifier for each trip
#  - **vendor_id** - a code indicating the provider associated with the trip record
#  - **pickup_datetime** - date and time when the meter was engaged
#  - **dropoff_datetime** - date and time when the meter was disengaged
#  - **passenger_count** - the number of passengers in the vehicle (driver entered value)
#  - **pickup_longitude** - the longitude where the meter was engaged
#  - **pickup_latitude** - the latitude where the meter was engaged
#  - **dropoff_longitude** - the longitude where the meter was disengaged
#  - **dropoff_latitude** - the latitude where the meter was disengaged
#  - **store_and_fwd_flag** - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server (Y=store and forward; N=not a store and forward trip)
#  - **trip_duration** - (target) duration of the trip in seconds
# 
# Here, we have 2 variables dropoff_datetime and store_and_fwd_flag which are not available before the trip starts and hence will not be used as features to the model.

# ### 1.1 Load Libraries

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from datetime import timedelta
import sklearn
import warnings
warnings.filterwarnings('ignore')



# ### Load Data

# In[ ]:


dfMain=pd.read_csv("D:/Internshala Course/EDA & ML module/EDA-ML-Final-Project/EDA+ML-Final Project/nyc_taxi_trip_duration.csv")

#df.head()
#copy_column=df[['pickup_latitude','pickup_longitude','dropoff_longitude','dropoff_latitude']].copy()
#copy_column.head()
#copy_column.to_csv("D:\Internshala Course\EDA & ML module", index= False)


# In[ ]:


df=dfMain.copy()


# In[ ]:





# In[ ]:


df.dtypes


# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe()


# ### File structure and content

# In[ ]:


# Convert the pickup and dropoff datetime columns to datetime format
df["pickup_datetime"]=pd.to_datetime(df["pickup_datetime"])
df["dropoff_datetime"]=pd.to_datetime(df["dropoff_datetime"])                                    


# ### Reformatting features & Checking consistency

# In[ ]:


df['store_and_fwd_flag'] = df['store_and_fwd_flag'].str.strip().map({'N': 0, 'Y': 1})
#df.head()


# In[ ]:


#print(df['pickup_datetime'].dtype)
#print(df['dropoff_datetime'].dtype)


# In[ ]:


# Create new columns for pickup date, pickup hour, pickup weekday, and trip duration in seconds
df['pickup_date']=df['pickup_datetime'].dt.date
df['pickup_day']=df['pickup_datetime'].dt.day
df['pickup_hour']=df['pickup_datetime'].dt.hour
df['pickup_weekday']=df['pickup_datetime'].dt.weekday
df['trip_duration']=df['trip_duration']/3600
df['dropoff_weekday']=df['dropoff_datetime'].dt.weekday
df['dropoff_day']=df['dropoff_datetime'].dt.day
#df['pickup_minute']=df['pickup_datetime'].dt.minute
#df['pickup_seconds']=df['pickup_datetime'].dt.second
df['total_time']=(df['dropoff_datetime']-df['pickup_datetime']).dt.total_seconds()
df['dropoff_hour']=df['dropoff_datetime'].dt.hour
df.shape


# In[ ]:


def time_of_day(x):
    # to calculate what time of it is now
    if x in range(6,12):
        return 'Morning'
    elif x in range(12,16):
        return 'Afternoon'
    elif x in range(16,22):
        return 'Evening'
    else:
        return 'Late night'

df['pickup_time_of_day'] = df['pickup_hour'].apply(time_of_day)
df['dropoff_time_of_day'] = df['dropoff_hour'].apply(time_of_day)


# In[ ]:


df.head()


# In[ ]:


# Define the dictionary mapping weekday numbers to weekday names
weekday_map1 = {0: 'pickup_Sunday', 1: 'pickup_Monday', 2: 'pickup_Tuesday', 3: 'pickup_Wednesday', 4: 'pickup_Thursday', 5: 'pickup_Friday', 6: 'pickup_Saturday'}
# Use the dt.weekday method to get the weekday number and map it to the corresponding weekday name
df['pickup_days'] = df['pickup_datetime'].dt.weekday.map(weekday_map1)



# In[ ]:


df.head()


# In[ ]:


# Define the dictionary mapping weekday numbers to weekday names
weekday_map2 = {0: 'dropoff_Sunday', 1: 'dropoff_Monday', 2: 'dropoff_Tuesday', 3: 'dropoff_Wednesday', 4: 'dropoff_Thursday', 5: 'dropoff_Friday', 6: 'dropoff_Saturday'}

# Use the dt.weekday method to get the weekday number and map it to the corresponding weekday name
df['dropoff_day'] = df['pickup_datetime'].dt.weekday.map(weekday_map2)



# In[ ]:


df.head()


# In[ ]:


check_trip_duration = (df['dropoff_datetime']-df['pickup_datetime']).dt.total_seconds()
df['duration_difference']= np.abs(check_trip_duration-df['trip_duration'])
duration_difference = df.query('duration_difference > 1')


# In[ ]:


duration_difference.shape


# ### Target Exploration

# In[ ]:


#df.head()
df['trip_duration'].describe()


# In[ ]:


df['log_trip_duration']=np.log10((df['trip_duration']*3600).values+1)


# In[ ]:


df.describe()


# In[ ]:


sns.distplot(df['log_trip_duration']).set(title='Distribution Plot with Log base 10 Transformation for Trip Duration')
# #plt.axvline(x=2.7, color="b")
plt.show()


# In[ ]:





# ## Univariate Visualization

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4),
                         gridspec_kw={'width_ratios': [4, 3, 3]})
#plt.figure(figsize=(33,10))
#passenger count plot
plt.subplot(131)
ax=sns.countplot(x='passenger_count',data=df, palette=sns.color_palette("husl"))
plt.xlabel('passenger count')
plt.ylabel('Frequency')
for p in ax.patches:
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black',fontsize=12)

#vendor_id plot
plt.subplot(132)
ax=sns.countplot(x='vendor_id', data=df, palette=sns.color_palette("husl"))
plt.xlabel("vendor id")
plt.ylabel("Frequency")
for p in ax.patches:
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black',fontsize=12)


#store_and_fwd_flag plot
plt.subplot(133)
ax=sns.countplot(x='store_and_fwd_flag', data=df, palette=sns.color_palette('husl'))
plt.xlabel("store and fwd flag")
plt.ylabel("Frequency")
for p in ax.patches:
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black',fontsize=12)


# In[ ]:





# Observations:
# 1. Most of the trips involve only 1 passenger. There are trips with 7-9 passengers but they are very low in number.
# 2. Vendor 2 has more number of trips as compared to vendor 1
# 3. The store_and_fwd_flag values, indicating whether the trip data was sent immediately to the vendor (“0”) or held in the memory of the taxi because there was no connection to the server (“1”), show that there was almost no storing taking place

# In[ ]:


df['pickup_datetime'].max(), df['pickup_datetime'].min()


# In[ ]:


df['dropoff_datetime'].max(), df['dropoff_datetime'].min()


# In[ ]:


#datetime plot
fig,axes=plt.subplots(nrows=1, ncols=2, figsize=(22,6),gridspec_kw={'width_ratios':[1,1]})
                
#weekday pickup count plot
ax=plt.subplot(121)
sns.countplot(x='pickup_weekday',data=df, palette=sns.color_palette('husl'))
plt.xlabel('week day')
plt.ylabel('Total Number of Pickups')
for p in ax.patches:
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')

#hour wise pickup count plot
ax=plt.subplot(122)
sns.countplot(x='pickup_hour',data=df, palette=sns.color_palette('husl'))
plt.xlabel('pickup_hour')
plt.ylabel('Total Number of Pickups')
for p in ax.patches:
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')


# * Number of pickups for weekends is much lower than week days with a peak on Thursday (4). Note that here weekday is a decimal number, where 0 is Sunday and 6 is Saturday.
# * Number of pickups as expected is highest in late evenings. However, it is much lower during the morning peak hours.

# In[ ]:





# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(12,5))
ax = df['vendor_id'].value_counts().plot(kind='bar',title="Vendors",ax=axes[0],color = ('blue',(1, 0.5, 0.13)))
df['vendor_id'].value_counts().plot(kind='pie',title="Vendors",ax=axes[1])
ax.set_ylabel("Count")
ax.set_xlabel("Vendor Id")
fig.tight_layout()


# In[ ]:





# In[ ]:


import folium
from folium.plugins import HeatMap

#Create map centered on New York City
nyc_coords = (40.712776, -74.005974)
m = folium.Map(location=nyc_coords, zoom_start=11,width="100%", height="100%")

#Create HeatMap using pickup latitude and longitude
heat_data = df[['pickup_latitude', 'pickup_longitude']].values.tolist()
HeatMap(heat_data, radius=8, max_zoom=13).add_to(m)

#Display map
m


# ### Lattitude & Longitude

# In[ ]:





# In[ ]:


with sns.plotting_context("notebook"):
            sns.set(style="white", palette=sns.color_palette("husl"), color_codes=True)
            f, axes = plt.subplots(2,2,figsize=(10, 10), sharex=False, sharey = False)
            sns.despine(left=True)
#creating 4 different plots on four different variable selected
            sns.histplot(df, x="pickup_latitude",  element="step", kde=True, stat="density",color="pink", common_norm=False, fill=False, bins=100, ax=axes[0,0])
            sns.histplot(df, x="pickup_longitude",  element="step", kde=True, stat="density", color="g",common_norm=False, fill=False, bins=100, ax=axes[1,0])
            sns.histplot(df, x="dropoff_latitude",  element="step", kde=True, stat="density", color="b",common_norm=False, fill=False, bins=100, ax=axes[0,1])
            sns.histplot(df, x="dropoff_longitude", element="step", kde=True, stat="density", color="r",common_norm=False, fill=False, bins=100, ax=axes[1,1])

            #plt.setp(axes, yticks=[])
            plt.tight_layout()
            plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


df.head()


# ### Bivariate Relations with Target
# Now that we have gone through all the basic features one by one. Let us start looking at their relation with the target. This will help us in selecting and extracting features at the modelling stage.

# ### Trip Duration vs Weekday

# In[ ]:


df.columns


# In[ ]:


# Setting the size of the plot

fig,axes=plt.subplots(nrows=1, ncols=2, figsize=(22,6),gridspec_kw={'width_ratios':[1,1]})
#selecting the spot between the two places to draw the plot
plt.subplot(121)
#Using median, we are creating the plot for pickup days with trip duration
summary_wdays_avg_duration = pd.DataFrame(df.groupby(['pickup_weekday'])['trip_duration'].median())
summary_wdays_avg_duration.reset_index(inplace = True)
summary_wdays_avg_duration['unit']=1

sns.set(style="white", palette="muted", color_codes=True)
sns.set_context("poster")
sns.lineplot(data=summary_wdays_avg_duration, x="pickup_weekday",units="unit", y="trip_duration")
sns.despine(bottom = False)

plt.subplot(122)
#Using median, we are creating the plot for pickup hours with trip duration
summary_hourly_avg_duration = pd.DataFrame(df.groupby(['pickup_hour'])['trip_duration'].median())
summary_hourly_avg_duration.reset_index(inplace = True)
summary_hourly_avg_duration['unit']=1

sns.set(style="white", palette="muted", color_codes=True)
sns.set_context("poster")
sns.lineplot(data=summary_hourly_avg_duration, x="pickup_hour", units = "unit", y="trip_duration")
sns.despine(bottom = False)


# In[ ]:





# ### Mean Trip Duration Vendor Wise

# In[ ]:


#setting the size of the plot and the ratio of two plots

fig,axes=plt.subplots(nrows=1, ncols=2, figsize=(22,6),gridspec_kw={'width_ratios':[1,1]})

plt.subplot(121)

# Generating plot with two variable on x-axis using mean

summary_wdays_avg_duration = pd.DataFrame(df.groupby(['vendor_id','pickup_weekday'])['trip_duration'].mean())
summary_wdays_avg_duration.reset_index(inplace = True)
summary_wdays_avg_duration['unit']=1

# Setting style for the plot

sns.set(style="white", palette="muted", color_codes=True)
sns.set_context("poster")
sns.lineplot(data=summary_wdays_avg_duration, x="pickup_weekday",hue="vendor_id",palette="Set2", y="trip_duration")
sns.despine(bottom = False)

# Generating the same plot as above, the difference only is this plot will be in reference to median

plt.subplot(122)
summary_wdays_avg_duration = pd.DataFrame(df.groupby(['vendor_id','pickup_weekday'])['trip_duration'].median())
summary_wdays_avg_duration.reset_index(inplace = True)
summary_wdays_avg_duration['unit']=1

sns.set(style="white", palette="muted", color_codes=True)
sns.set_context("poster")
sns.lineplot(data=summary_wdays_avg_duration, x='pickup_weekday', units = "unit",palette='Set2', hue="vendor_id", y="trip_duration")
sns.despine(bottom =False)



# ### Trip Duration vs Passenger Count

# In[ ]:


# couting the trips with respect to number of passenger in each trip to find outlier
df.passenger_count.value_counts()


# In[ ]:


# Generating box plot with respect to passenger_count and trip_duration

df.passenger_count.value_counts()

plt.figure(figsize=(22, 6))

df_sub = df[df['trip_duration']*3600 < 10000]

sns.boxplot(x="passenger_count", y="trip_duration", data=df_sub)

plt.show()



# ### The relationship between vendor id and duration

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Generating line plot for pickout_hour

group1 = df.groupby('pickup_hour').trip_duration.mean().to_frame().reset_index()

plt.figure(figsize=(10,6))

sns.pointplot(x='pickup_hour', y='trip_duration', data=group1)

plt.ylabel('Trip Duration (hour)')

plt.xlabel('Pickup Hour')

plt.show()


# In[ ]:





# ### Pick Up Points v/s Dropoff Points

# In[ ]:


df.columns


# In[ ]:


city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,figsize = (12,5))
ax[0].scatter(df['pickup_longitude'].values, df['pickup_latitude'].values,
color='blue', s=1, label='train', alpha=0.1)
ax[1].scatter(df['dropoff_longitude'].values, df['dropoff_latitude'].values,
color='green', s=1, label='train', alpha=0.1)
ax[1].set_title('Drop-off Co-ordinates')
ax[0].set_title('Pick-up Co-ordinates')
ax[0].set_ylabel('Latitude')
ax[0].set_xlabel('Longitude')
ax[1].set_ylabel('Latitude')
ax[1].set_xlabel('Longitude')
plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.show()


# In[ ]:





# In[ ]:





# ## Evaluation Metric (MAE)
# - A suitable evaluation metric would be Mean Absolute Error (MAE). MAE is the average of the absolute differences between the predicted and actual values. This metric is preferred because it gives equal weightage to all errors and is less sensitive to outliers compared to other metrics such as Root Mean Squared Error (RMSE).

# ### Bencmark Model

# In[ ]:


dfben=df.copy()


# In[ ]:


from sklearn.utils import shuffle
dfben= shuffle(dfben, random_state=10)


# In[ ]:


# diividing dataset in 3:1 ratio

div=int(df.shape[0]/4)

Train=dfben.loc[:3*div+1,:]
Test= dfben.loc[3*div+1:]


# In[ ]:


#Train.head()


# In[ ]:


Test.head()


# In[ ]:


# Taking mean of trip_duration  column from Train part of dataset

Test['mean_trip_duration']=Train['trip_duration'].mean()
#Test.head()


# In[ ]:


from sklearn.metrics import mean_absolute_error as MAE
# Finding Mean Absolute Error 
simple_mean_error=MAE(Test['trip_duration'],Test['mean_trip_duration'])
simple_mean_error


# ### mean error for pickup_day with respect to mean_trip_duration

# In[ ]:


# Making a pivot table for particular columns
week_day=pd.pivot_table(Train, values='trip_duration', index=['pickup_weekday'], aggfunc=np.mean)
week_day


# In[ ]:


# Applying for loop in order to store unique values in a given column
Test['week_day_mean']=0

for i in Train['pickup_weekday'].unique():
# Assigning the unique value to a new created column after posing some operations on train dataset

    Test['week_day_mean'][Test['pickup_weekday']==str(i)]=Train['trip_duration'][Train['pickup_weekday']==str(i)].mean()


# In[ ]:


# Againg finding Mean Absolute Error
pickup_weekday_error=MAE(Test['trip_duration'],Test['week_day_mean'])
pickup_weekday_error


# #### Mean trip_duration_hour with respect to passenger_count

# In[ ]:


#trip_duration_hour mean with respect to the mean of passenger_count

pass_count = pd.pivot_table(Train, values='trip_duration', index = ["passenger_count"], aggfunc=np.mean)
pass_count


# In[ ]:


# Applying for loop to collect unique values
Test['pass_count_mean']=0
for i in Train['passenger_count'].unique():
    
    Test['pass_count_mean'][Test['passenger_count']==str(i)]=Train['trip_duration'][Train['passenger_count']==str(i)].mean()
    


# In[ ]:


pass_count_error=MAE(Test['trip_duration'],Test['pass_count_mean'])
pass_count_error


# ####  pickup_day vs trip_duration

# In[ ]:


pick_up=pd.pivot_table(Train, values='trip_duration',index='pickup_day', aggfunc=np.mean)
pick_up


# In[ ]:


Test['pick_mean']=0
for i in Train['pickup_day'].unique():
    Test['pick_mean'][Test['pickup_day']==str(i)]=Train['trip_duration'][Train['pickup_day']==str(i)].mean()
    


# In[ ]:





# In[ ]:


pickup_day_error=MAE(Test['trip_duration'],Test['pick_mean'])
pickup_day_error


# In[ ]:


Test.head()
#Test.shape
#Test.describe()


# ### Mean trip_duration_hour with respect to passenger_count, pickup_time_of_day and dropoff_time_of_day

# In[ ]:


combo = pd.pivot_table(Train, values = 'trip_duration', index = ['passenger_count','pickup_time_of_day','dropoff_time_of_day'], aggfunc = np.mean)
combo


# In[ ]:





# In[ ]:


# Define the three columns to group by
group_cols = ['passenger_count', 'pickup_time_of_day', 'dropoff_time_of_day']

# Calculate the mean of trip_duration for each group in the train dataset
train_group_means = Train.groupby(group_cols)['trip_duration'].transform('mean')

# Assign the mean value to the Super_mean column in the test dataset
Test['Super_mean'] = train_group_means


# In[ ]:


Test.dropna(subset=['Super_mean'], inplace=True)


# In[ ]:


#calculating mean absolute error
super_mean_error = MAE(Test['trip_duration'] , Test['Super_mean'] )
super_mean_error


# ### Mean trip_duration_hour with respect store_and_fwd_flag

# In[ ]:


flag=pd.pivot_table(Train, values='trip_duration', index='store_and_fwd_flag', aggfunc=np.mean)
flag


# In[ ]:


Test['Flag_mean']=0

for i in Train['store_and_fwd_flag'].unique():
    
    Test['Flag_mean'][Test['store_and_fwd_flag']==str(i)]=Train['trip_duration'][Train['store_and_fwd_flag']==str(i)].mean()
    


# In[ ]:


flag_mean_error=MAE(Test['trip_duration'],Test['Flag_mean'])
flag_mean_error


# ### Conclusion

# 1) The error of simple mean of trip duration hour is 0.17277651586027712 which is also almost equal to MAE of dropp time and     pickup time of the day is 0.17263892934259373, 0.17262796487353846 respectively
# 
# 2) The str_fwd_error and passanger count error is same i.e. 0.2652592807074405
# 
# 3) The mean of trip_duration_hour with respect to pickup_time_error , dropoff_time_error and pass_count error is      0.16813712281281915

# # KNN Analysis

# In[ ]:


# Create a copy of initial dataset df so that it doesn't get modified too much
dfmath=df.copy()


# In[ ]:


#dfmath.shape
#df.describe()
dfmath.columns


# In[ ]:





# In[ ]:


import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the earth in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_km = R * c
    return distance_km

# Calculate the distance for each row in the dataset
dfmath['distance_km'] = haversine(dfmath['pickup_latitude'], dfmath['pickup_longitude'], dfmath['dropoff_latitude'], dfmath['dropoff_longitude'])
print(dfmath['distance_km'].describe())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


dfknn=dfmath.copy()


# In[ ]:





# In[ ]:


# pd.options.display.float_format = '{:.6f}'.format
# df['trip_duration']
#df.head()
#
dfknn.columns


# In[ ]:


print(dfknn['distance_km'].describe())


# In[ ]:


# Creating a scatter plot inorder to identify outlier
plt.figure(figsize=(8, 6))
a=dfknn['trip_duration']
ax=sns.scatterplot(data=dfknn, x=a, y="distance_km", s=50, palette='viridis', marker="s")
a.describe()


# In[ ]:


dfknn[a>500]


# In[ ]:


dfknn[dfknn['distance_km']>300]


# In[ ]:


#Dropping the value from the dataset
dfknn.drop(dfknn[dfknn['id'].isin(['id1864733', 'id2306955','id0982904','id2644780','id0116374','id0978162	'])].index, inplace=True)


# In[ ]:


plt.figure(figsize=(8, 6))
sns.scatterplot(data=dfknn, x=a, y="distance_km", s=50, palette='viridis', marker="s")


# In[ ]:


# Binary Features
plt.figure(figsize=(22, 6))
#fig, axs = plt.subplot(ncols=2)

# Passenger Count
plt.subplot(121)
ax=sns.countplot(data=dfknn, x='passenger_count',palette=sns.color_palette('husl'))
plt.xlabel('Passenger Count')
plt.ylabel('Frequency')
for p in ax.patches:
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black')


# In[ ]:


dfknn = dfknn.loc[~dfknn['passenger_count'].isin([0, 7, 9])]


# In[ ]:


# Binary Features
plt.figure(figsize=(22, 6))
#fig, axs = plt.subplot(ncols=2)

# Passenger Count
plt.subplot(131)
ax=sns.countplot(data=df, x='passenger_count',palette=sns.color_palette('husl'))
plt.xlabel('Passenger Count')
plt.ylabel('Frequency')
for p in ax.patches:
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
                    ha='center', va='bottom',
                    color= 'black',fontsize=15)


# In[ ]:


import seaborn as sns
import pandas as pd

# Load the dataset

# Compute the correlation matrix
corr = dfknn.corr()
fig, ax = plt.subplots(figsize=(35, 35))
# Create a heatmap to visualize the correlation matrix
sns.heatmap(corr, annot=True, cmap='coolwarm')


# In[ ]:


import seaborn as sns
import pandas as pd

# Load the dataset

# Compute the correlation matrix
corr = Test.corr()
fig, ax = plt.subplots(figsize=(35, 35))
# Create a heatmap to visualize the correlation matrix
sns.heatmap(corr, annot=True, cmap='coolwarm')


# In[ ]:


dfknn.dtypes


# In[ ]:


#d=dfknn
#d.drop(columns=['pickup_day','dropoff_day','weekday_name''pickup_day','dropoff_day'],inplace=True)
#d.drop(columns=['id','vendor_id','pickup_datetime','dropoff_datetime','pickup_date','dropoff_weekday','weekday_name','pickup_weekday'],inplace=True)
#data.dtypes
#print(type(df.loc[1, "id"]))
#d.dtypes


# In[ ]:


data = dfknn.iloc[1:18001,]
cat_cols = ['pickup_time_of_day', 'dropoff_time_of_day']
data = pd.concat([data, pd.get_dummies(data[cat_cols].astype('str'))], axis=1)
data.drop(columns = ['pickup_time_of_day','pickup_datetime','dropoff_datetime', 'dropoff_time_of_day','id','vendor_id','pickup_date','pickup_days','dropoff_day'],inplace = True)
data


# In[ ]:


#seperate features and target
x = data.drop(['trip_duration'], axis=1)
y = data["trip_duration"]
x.shape,y.shape


# In[ ]:


#dfknn.drop(columns=['pickup_days'],inplace=True)


# In[ ]:


# Importing MinMax Scaler

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)


# In[ ]:


x = pd.DataFrame(x_scaled)


# In[ ]:


# Importing Train test split
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y, random_state = 56)


# In[ ]:


#importing KNN regressor and metric mse
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.metrics import mean_absolute_error as MAE


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Creating instance of KNN
reg = KNN(n_neighbors = 5)

# Fitting the model
reg.fit(train_x, train_y)

# Predicting over the Train Set and calculating MSE
test_predict = reg.predict(test_x)
k = MAE(test_predict, test_y)



mse = mean_squared_error(test_y,test_predict,squared=False)
rmse = np.sqrt(mse)
r2 = r2_score(test_y,test_predict)

print("RMSE:       ", rmse)
print("R2:         ", r2)

print('Test MAE    ', k )



# In[ ]:


def Elbow(K):
  #initiating empty list
    test_MAE = []
  
  #training model for evey value of K
    for i in K:
        #Instance of KNN
        reg = KNN(n_neighbors = i)
        reg.fit(train_x, train_y)
        #Appending MAE value to empty list claculated using the predictions
        tmp = reg.predict(test_x)
        tmp = MAE(tmp,test_y)
        test_MAE.append(tmp)
    
    return test_MAE


# In[ ]:


#Defining K range
k = range(1,10)


# In[ ]:


test=Elbow(k)


# In[ ]:


# plotting the Curves
plt.plot(k, test)
plt.xlabel('K Neighbors')
plt.ylabel('Test Mean Absolute Error')
plt.title('Elbow Curve for test')


# In[ ]:


knn_train_score = reg.score(train_x,train_y)
print(knn_train_score*100,"%")


# In[ ]:





# In[ ]:


knn_test_score = reg.score(test_x,test_y)
knn_test_score*100


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Creating instance of KNN
reg = KNN(n_neighbors = 3)

# Fitting the model
reg.fit(train_x, train_y)

# Predicting over the Train Set and calculating MSE
test_predict = reg.predict(test_x)
k = MAE(test_predict, test_y)

mse = mean_squared_error(test_y,test_predict,squared=False)
rmse = np.sqrt(mse)
r2 = r2_score(test_y,test_predict)


knn_train_score = reg.score(train_x,train_y)

knn_test_score = reg.score(test_x,test_y)

print("Knn_test_score:  ", knn_test_score)

print("knn_train_score: ",knn_train_score)
print("RMSE:            ", rmse)
print("R2:              ", r2)


print('Test MAE:        ', k )


# In[ ]:





# In[ ]:





# ### Observation
# 1) Preprocessing the columns 'pickup_time_of_day' and 'dropoff_time_of_day' and divide them into 4 parts. using general method of finding the error in KNN algorithm, the MAE comes out to be as follows
# 2) For the features used in above calculations, we got a bit under fit results. Although the results are not that bad.
# 3) for K=5, we ha R2 value of 0.813186163341647 and for K=3, we have R2 value of  0.9135591687208122. 
# 4) These values are pretty impressive but there is a big room for improvement in the hypothesis.

# ### KNN analysis for multiple features

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Select the top 5 features
X = dfknn[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'distance_km']]
y = dfknn['trip_duration']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the KNN regressor
knn = KNeighborsRegressor()

# Fit the model on the training data
knn.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = knn.predict(X_test)

# Evaluate the model performance using mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error:', mse)

r2 = r2_score(y_test, y_pred)
print('R-squared:', r2)

k = MAE(y_pred, y_test)
print('Test MAE:          ', k )


# In[ ]:





# # Linear Regression

# In[ ]:


dflr=df.copy()


# In[ ]:


import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the earth in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_km = R * c
    return distance_km

# Calculate the distance for each row in the dataset
dflr['distance_km'] = haversine(dflr['pickup_latitude'], dflr['pickup_longitude'], dflr['dropoff_latitude'], dflr['dropoff_longitude'])


# In[ ]:


dflr.shape


# In[ ]:





# In[ ]:


dflr.head()


# #### Linear Regression on ('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',   'distance_km') features

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Select the top 5 features
X = dflr[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'distance_km']]
y = dflr['trip_duration']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the linear regression model
lr = LinearRegression()

# Fit the model on the training data
lr.fit(X_train, y_train)

train_predict=lr.predict(x_train)
k1 = mae(train_predict, train_y)
print('Training Mean Absolute Error', k1 )

# Make predictions on the testing data
y_pred = lr.predict(X_test)

# Evaluate the model performance using metrics such as mean squared error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
k = MAE(y_pred, y_test)
print('Test MAE:          ', k )
print('Mean squared error:', mse)
print('R-squared:', r2)


# #### Linear Regression on ('distance_km','pickup_hour','passenger_count') features

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


X = dflr[['distance_km','pickup_hour','passenger_count']]
y = dflr['trip_duration']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the linear regression model
lr = LinearRegression()

# Fit the model on the training data
lr.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = lr.predict(X_test)

# Calculate the mean squared error and R-squared value
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

k = MAE(y_pred, y_test)
print('Test MAE:          ', k )
print('Mean squared error:', mse)
print('R-squared:', r2)


# #### Linear Regression on preprocessed columns(days of the week for pickup and dropoff)

# In[ ]:


dflr.head()


# In[ ]:


# Selecting a part of dataset for train dataset 
data = dflr.iloc[1:18001,]
cat_cols = ['pickup_days', 'dropoff_day']
# Creating dummy columns
data = pd.concat([data, pd.get_dummies(data[cat_cols].astype('str'))], axis=1)
data.drop(columns = ['pickup_days', 'dropoff_day','id','vendor_id','pickup_time_of_day','dropoff_time_of_day','pickup_datetime','dropoff_datetime','pickup_date'],inplace = True)
data


# In[ ]:


data.dtypes


# In[ ]:


#seperating independent and dependent variables
x = data.drop(['trip_duration'], axis=1)
y = data['trip_duration']
x.shape, y.shape


# In[ ]:


# Performing the train test split function
train_x,test_x,train_y,test_y = train_test_split(x,y, random_state = 56)


# In[ ]:


#importing Linear Regression 
from sklearn.linear_model import LinearRegression as LR


# In[ ]:





# In[ ]:


dflr.dtypes


# In[ ]:


# Creating instance of Linear Regresssion
lr = LR()

# Fitting the model
lr.fit(train_x, train_y)


# In[ ]:


# Predicting over the Train Set and calculating error
train_predict = lr.predict(train_x)
k = MAE(train_predict, train_y)


from sklearn.metrics import mean_squared_error, r2_score

# Calculate RMSE
rmse = mean_squared_error(train_y, train_predict, squared=False)
print('RMSE:', rmse)


print('Training Mean Absolute Error', k )


# In[ ]:


# Predicting over the Test Set and calculating error
test_predict = lr.predict(test_x)
k = MAE(test_predict, test_y)


from sklearn.metrics import mean_squared_error, r2_score

# Calculate RMSE
rmse = mean_squared_error(test_y, test_predict, squared=False)
print('RMSE:', rmse)
k = MAE(test_predict, test_y)
print('Test MAE:          ', k )

print('Test Mean Absolute Error    ', k )


# #### Parameters of Linear Regression

# In[ ]:


lr.coef_


# #### Plotting the coefficients

# In[ ]:


range(len(train_x.columns))


# In[ ]:


plt.figure(figsize=(8, 6), dpi=120, facecolor='w', edgecolor='b')
x = range(len(train_x.columns))
y = lr.coef_
plt.bar( x, y )
plt.xlabel( "Variables")
plt.ylabel('Coefficients')
plt.title('Coefficient plot')


# #### Checking assumptions of Linear Model

# In[ ]:


# Arranging and calculating the Residuals
residuals = pd.DataFrame({
    'fitted values' : test_y,
    'predicted values' : test_predict,
})

residuals['residuals'] = residuals['fitted values'] - residuals['predicted values']
residuals.head()


# #### Checking Distribution of Residuals

# In[ ]:


plt.figure(figsize=(10,6),dpi=150,facecolor="w",edgecolor="b")
plt.hist(residuals.residuals,bins=100)
plt.xlabel("error")
plt.ylabel("frequency")
plt.title("distribution of error terms")
plt.show()


# #### QQ-Plot

# In[ ]:


# importing the QQ-plot from the from the statsmodels
from statsmodels.graphics.gofplots import qqplot

## Plotting the QQ plot
fig, ax = plt.subplots(figsize=(5,5) , dpi = 120)
qqplot(residuals.residuals, line = 's' , ax = ax)
plt.ylabel('Residual Quantiles')
plt.xlabel('Ideal Scaled Quantiles')
plt.legend(["Residual Quantiles","Ideal Scaled Quantiles"])
plt.title('Checking distribution of Residual Errors')
plt.show()


# In[ ]:


# Importing Variance_inflation_Factor funtion from the Statsmodels
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Calculating VIF for every column (only works for the not Catagorical)
VIF = pd.Series([variance_inflation_factor(data.values, i) for i in range(data.shape[1])], index =data.columns)
VIF


# #### Model Interpretability

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Create an instance of StandardScaler
scaler = StandardScaler()

# Normalize the input features
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

# Create an instance of Linear Regression
lr = LinearRegression()

# Fit the model
lr.fit(train_x_scaled, train_y)


# In[ ]:


# Predicting over the Train Set and calculating error
train_predict = lr.predict(train_x)
k = MAE(train_predict, train_y)
print('Training Mean Absolute Error', k )


# In[ ]:


# Predicting over the Test Set and calculating error
test_predict = lr.predict(test_x)
k = MAE(test_predict, test_y)


from sklearn.metrics import mean_squared_error, r2_score

# Calculate RMSE
rmse = mean_squared_error(test_y, test_predict, squared=False)
print('RMSE:', rmse)



print('Test Mean Absolute Error    ', k )


# In[ ]:


plt.figure(figsize=(8, 6), dpi=120, facecolor='w', edgecolor='b')
x = range(len(train_x.columns))
y = lr.coef_
plt.bar( x, y )
plt.xlabel( "Variables")
plt.ylabel('Coefficients')
plt.title('Normalized Coefficient plot')


# # Conclusions
# 1) We have done regression on 4 different set of features. Unfortunately, no any hypothesis gave us very good results.
# 2) Our Linear regression model will fall under the catagory of either under fit or over fit.
# 3) On computing the coefficients we observed that there are some negative values as well
# 4) On plotting the qqplot we see that the residual quantile line doesn't fit over all ideal scaled quantiles

# ## Comparison of best MAE for three models
# Note: This is not a best way to comparing the models as we are comparing only MAE.
# 
# * Other parameters are more important determining if the model is good or not like, R2, RMSE, MSE.
# * This is just a general comparison of three models because our model didn't perform well in linear regression.

# In[ ]:


import matplotlib.pyplot as plt

mae_benchmark = 0.0961111111111111
mae_knn =  0.06723621399176954
mae_linear =   0.12728821350623243

models = ['Benchmark', 'KNN', 'Linear Regression']
mae_values = [mae_benchmark, mae_knn, mae_linear]

plt.bar(models, mae_values, color=['red', 'green', 'blue'])
plt.title('Comparison of MAE Values for Three Models')
plt.xlabel('Models')
plt.ylabel('MAE')

for i in range(len(models)):
    plt.annotate(str(round(mae_values[i], 3)), xy=(models[i], mae_values[i]), ha='center', va='bottom')

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




