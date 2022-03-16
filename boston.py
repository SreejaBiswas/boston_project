# import yfinance as yf

# from fbprophet.plot import plot_plotly
# from plotly import graph_objs as go
from cProfile import label
from re import A
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


st.title(' *Linear Regression on Boston House Prices* ')
st.write('This project will predict the *Boston House Price*..')
from PIL import Image
img = Image.open("boston.png")
st.image(img)

st.subheader("Introduction")
st.write("*In this project, we will develop and evaluate the performance and the predictive power of a model trained and tested on data collected from houses in Boston’s suburbs.Once we get a good fit, we will use this model to predict the monetary value of a house located at the Boston’s area.A model like this would be very valuable for a real state agent who could make use of the information provided in a dayly basis.*")





dataset=load_boston()
df=pd.DataFrame(dataset.data)
df.columns=dataset.feature_names
df["PRICES"]=dataset.target


from PIL import Image
img = Image.open("boston4.webp")
st.sidebar.image(img)


st.sidebar.write("*Do you want to see the dataset?*")
if st.sidebar.checkbox(' View dataset in table data format '):
    st.dataframe(df)
# Show each column description when checkbox is ON.
st.sidebar.write("*To know about the columns click below!*")


from PIL import Image
img = Image.open("boston3.jpg")
st.image(img)


st.text("The features can be summarized as follows:")
if st.sidebar.checkbox('Show each column name and its description'):
    st.markdown(
 r"""
 ## *Column name and its Description*
 #### CRIM: Crime occurrence rate per unit population by town
 #### ZN: Percentage of 25000-squared-feet-area house
 #### INDUS: Percentage of non-retail land area by town
 #### CHAS: Index for Charlse river: 0 is near, 1 is far
 #### NOX: Nitrogen compound concentration
 #### RM: Average number of rooms per residence
 #### AGE: Percentage of buildings built before 1940
 #### DIS: Weighted distance from five employment centers
 #### RAD: Index for easy access to highway
 #### TAX: Tax rate per 100,000 dollar
 #### PTRATIO: Percentage of students and teachers in each town
 #### B: 1000(Bk - 0.63)^2, where Bk is the percentage of Black people
 #### LSTAT: Percentage of low-class population
 ####
 """
 )
 # Plot the relation between target and explanatory variables
# when the checkbox is ON.
st.sidebar.write("*Showing relation..*")
if st.sidebar.checkbox('Plot the relation between target and explanatory variables'):
    # Select one explanatory variable for ploting
    checked_variable = st.selectbox(
    'Select one explanatory variable:',
    df.drop(columns="PRICES").columns
    )
    # Plot
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(x=df[checked_variable], y=df["PRICES"])
    plt.xlabel(checked_variable)
    plt.ylabel("PRICES")
    st.pyplot(fig)




"""
## *Preprocessing*

"""
# Select the variables you will NOT use
Features_chosen = []
Features_NonUsed = st.multiselect(
 'Select the variables you will NOT use',
 df.drop(columns="PRICES").columns
 )
# Drop the columns you selected
df = df.drop(columns=Features_NonUsed)
# Choose whether you will perform logarithmic transformation
left_column, right_column = st.columns(2)
bool_log = left_column.radio(
 'You will perform logarithmic transformation?',
 ('No','Yes')
 )
df_log, Log_Features = df.copy(), []
if bool_log == 'Yes':
 Log_Features = right_column.multiselect(
 'Select the variables you will perform logarithmic transformation',
 df.columns
 )
 # Perform the lagarithmic transformation
 df_log[Log_Features] = np.log(df_log[Log_Features])
 # Choose whether you will perform standardization
left_column, right_column = st.columns(2)
bool_std = left_column.radio(
 'You will perform standardization?',
 ('No','Yes')
 )
df_std = df_log.copy()
if bool_std == 'Yes':
 Std_Features_NotUsed = right_column.multiselect(
 'Select the variables you will NOT perform standardization',
 df_log.drop(columns=["PRICES"]).columns
 )
 # Assign the explanatory variables,
 # excluded of ones in "Std_Features_NotUsed",
 # to "Std_Features_chosen"
 Std_Features_chosen = []
 for name in df_log.drop(columns=["PRICES"]).columns:
    if name in Std_Features_NotUsed:
        continue
    else:
        Std_Features_chosen.append(name)
 # Perform standardization
 sscaler = preprocessing.StandardScaler()
 sscaler.fit(df_std[Std_Features_chosen])
 df_std[Std_Features_chosen] = sscaler.transform(df_std[Std_Features_chosen])


from PIL import Image
img = Image.open("boston2.png")
st.image(img)


"""
## *Split the dataset into training and validation datasets*
"""
left_column,right_column = st.columns(2)
l1=left_column.slider('Validation Data Size',0.01,0.99)
left_column.text('Selected: {}'.format(l1))
r1=right_column.slider('Non-negative Interger',0)
right_column.text('Selected: {}'.format(r1))

# Split the dataset
X_train, X_val, Y_train, Y_val = train_test_split(
 df_std.drop(columns=["PRICES"]),
 df_std['PRICES'],
 test_size=l1,
 random_state=r1
)

 # Create an instance of liner regression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
Y_pred_train = regressor.predict(X_train)
Y_pred_val = regressor.predict(X_val)
# Perform inverse conversion if the logarithmic transformation was performed.
if "PRICES" in Log_Features:
 Y_pred_train, Y_pred_val = np.exp(Y_pred_train), np.exp(Y_pred_val)
 Y_train, Y_val = np.exp(Y_train), np.exp(Y_val)
st.sidebar.write("*Finally results!!*")


"""
    ## *Accuracy of the model*
"""

R2=r2_score(Y_val, Y_pred_val)
if st.button("Show the value of R2"):
     st.write(f'R2 value : {R2:.2f}')

from PIL import Image
img = Image.open("boston6.jpg")
st.sidebar.image(img)

from PIL import Image
img = Image.open("boston7.jpg")
st.image(img)

"""
## *Plot the results*
"""
left_column, right_column = st.columns(2)
show_train = left_column.radio(
 'Plot the result of the training dataset:',
 ('Yes','No')
 )
show_val = right_column.radio(
 'Plot the result of the validation dataset:',
 ('Yes','No')
 )

# default axis range
y_max_train = max([max(Y_train), max(Y_pred_train)])
y_max_val = max([max(Y_val), max(Y_pred_val)])
y_max = int(max([y_max_train, y_max_val])) 
# interactive axis range
left_column, right_column = st.columns(2)
x_min = left_column.number_input('x_min:',value=0,step=1)
x_max = right_column.number_input('x_max:',value=y_max,step=1)
left_column, right_column = st.columns(2)
y_min = left_column.number_input('y_min:',value=0,step=1)
y_max = right_column.number_input('y_max:',value=y_max,step=1)

fig = plt.figure(figsize=(3, 3))
if show_train == 'Yes':
	plt.scatter(Y_train, Y_pred_train,lw=0.1,color="b",label="Training Data")
if show_val == 'Yes':
	plt.scatter(Y_val, Y_pred_val,lw=0.1,color="r",label="Validation Data")
plt.xlabel("PRICES",fontsize=8)
plt.ylabel("PRICES of prediction",fontsize=8)
plt.xlim(int(x_min), int(x_max)+5)
plt.ylim(int(y_min), int(y_max)+5)
plt.legend(fontsize=6)
plt.tick_params(labelsize=6)
st.pyplot(fig)


st.subheader("Conclusion")
st.write("*Throughout this article we made a machine learning regression project from end-to-end and we learned and obtained several insights about regression models and how they are developed.*")

    
    


