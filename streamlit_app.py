#%%
###import packages


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
import streamlit.components.v1 as components
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
import random

#%%
st.set_page_config(layout="wide")


# Text/Title
st.title("Logistic Regression - Mnist Dataset")

#%%
#Navigation bar
with st.beta_container():
 #navbar 
#https://bootsnipp.com/snippets/nNX3a     https://www.mockplus.com/blog/post/bootstrap-navbar-template
   components.html(
       """
       <link href="//maxcdn.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css" rel="stylesheet" id="bootstrap-css">
<script src="//maxcdn.bootstrapcdn.com/bootstrap/4.1.1/js/bootstrap.min.js"></script>
<script src="//cdnjs.cloudflare.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
<!------ Include the above in your HEAD tag ---------->
<nav class="navbar navbar-icon-top navbar-expand-lg navbar-dark bg-dark" >
  <a class="navbar-brand" href="https://www.rstiwari.com" target="_blank">Profile</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarSupportedContent">
    <ul class="navbar-nav mr-auto">
      <li class="nav-item active">
        <a class="nav-link" href=" https://tiwari11-rst.medium.com/" target="_blank" >
          <i class="fa fa-home"></i>
          Medium
          <span class="sr-only">(current)</span>
          </a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href=" https://happyman11.github.io/" target="_blank">
          <i class="fa fa-envelope-o">
            <span class="badge badge-danger">Git Pages</span>
          </i>
         
        </a>
      </li>
      
        <li class="nav-item">
        <a class="nav-link" href="https://happyman11.github.io/" target="_blank">
          <i class="fa fa-globe">
            <span class="badge badge-success">Badges</span>
          </i>
         
        </a>
      </li>
          
        </a>
      </li>
      
      <li class="nav-item">
        <a class="nav-link disabled" href="https://ravishekhartiwari.blogspot.com/" target="_blank">
          <i class="fa fa-envelope-o">
            <span class="badge badge-warning">Blogspot</span>
          </i>
          
        </a>
      </li>
      
      
    </ul>
  
    
  </div>
</nav>
       """, height=70,
    )



###
st.sidebar.markdown("""** Hyperparameter Selection **""")
Split = st.sidebar.slider('Train-Test Splitup (in %)', 0.1,0.9,.70)
st.sidebar.markdown("""**Select SVM Parameters**""")

kernal= st.sidebar.selectbox('Kernel',('rbf','linear', 'poly', 'sigmoid', 'precomputed'))

Tol= st.sidebar.text_input("Tolerance for stopping Criteria (default: 1e-3)","1e-3")
Max_Iteration=st.sidebar.text_input("Number of iteration (default: -1)","-1")


#%%
#creating dictionary for hyperparameters
parameters={ 'Kernal':kernal,
             'Tol':Tol,
             'Max_Iteration':Max_Iteration,
   	     'kernal':kernal
              }
       

#%%

#%%

#Functions
#Function for the dataset load
def load_dataset():
    data=sklearn.datasets.load_digits()
    return data

#select dataset

def select_dataset(data):

    random_no=random.randint(0,len(data.images))
    image=data.images[random_no]
    label=data.target[random_no]
    return(image,label)

    
def display_dataset(dataset):

    image,label=select_dataset(dataset)
    st.write("Label :",label)
    fig, ax = plt.subplots()
    
    ax.imshow(image,cmap='inferno')
    st.pyplot(fig)
        
def train_test_splited(data,split):

    n_samples = len(data.images)
    dataset_flat = data.images.reshape((n_samples, -1))
    X_train, X_test, y_train, y_test = train_test_split(dataset_flat,data.target, test_size=float(split),
    random_state=42)

    return ( X_train, X_test, y_train, y_test)

def SVM_model(parameters,Data):

    X_train, X_test, y_train, y_test= train_test_splited(Data,Split)

    clf=svm.SVC(kernel=parameters['kernal'],tol=float(parameters['Tol']),max_iter=int(parameters['Max_Iteration']))
    clf=clf.fit(X_train,y_train)
    
    model_data={"model":clf,
                 "Y_actual": y_test,
                  "X_test": X_test }

    return(model_data)

def prediction_test(model):

     prediction=model["model"].predict(model["X_test"])
     return(prediction)

def prediction_plot(model,prediction):
    
   
    random_no=random.randint(0,len(prediction))
    image=model["X_test"][random_no]
    st.write(" Prediction :",prediction[random_no],"Actual Label :",model["Y_actual"][random_no] )
    fig, ax = plt.subplots()
    
    image = image.reshape(8, 8)
    ax.imshow(image,cmap='inferno')
    st.pyplot(fig)

    
    
#main
dataset=load_dataset()

with st.spinner('Loading Dataset..'):
     time.sleep(2)

st.subheader('Mnist-Dataset')
col1, col2, col3 = st.beta_columns(3)
with col1:
    st.write("***Dataset***: Classification")
with col2:
    st.write("***DIgits***:0-9")
with col3:
    st.write("***Shape***: 8*8 matrix")
    


col1, col2, col3,col4,col5 = st.beta_columns(5)
with st.spinner('Loading Dataset..'):
     time.sleep(2)


with col1:
    display_dataset(dataset)
     
with col2:
    display_dataset(dataset)

with col3:
    display_dataset(dataset)
     
with col4:
    display_dataset(dataset)

with col5:
    display_dataset(dataset)

col6, col7, col8,col9,col10 = st.beta_columns(5)
with st.spinner('Loading Dataset..'):
     time.sleep(2)

with col6:
    display_dataset(dataset)
     
with col7:
    display_dataset(dataset)

with col8:
    display_dataset(dataset)
     
with col9:
    display_dataset(dataset)

with col10:
    display_dataset(dataset)

col11, col12, col13,col14,col15 = st.beta_columns(5)
with st.spinner('Loading Dataset..'):
     time.sleep(2)

with col11:
    display_dataset(dataset)
     
with col12:
    display_dataset(dataset)

with col13:
    display_dataset(dataset)
     
with col14:
    display_dataset(dataset)

with col15:
    display_dataset(dataset)


col16, col17, col18,col19,col20 = st.beta_columns(5)
with st.spinner('Loading Dataset..'):
     time.sleep(2)

with col16:
    display_dataset(dataset)
     
with col17:
    display_dataset(dataset)

with col18:
    display_dataset(dataset)
     
with col19:
    display_dataset(dataset)

with col20:
    display_dataset(dataset)


if(st.sidebar.button("Click to train the SVN Classification Model")):
    
    my_bar = st.progress(0)
    for percent_complete in range(100):
        
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1)
        
    with st.spinner('Trainning.....'):
        time.sleep(2)

   
    model=SVM_model(parameters,dataset)
    prediction=prediction_test(model)
    
    st.subheader("Prediction By Model on Test Data")
    
    col21, col22, col23,col24,col25 = st.beta_columns(5)
    

    with col21:
        prediction_plot(model,prediction)
     
    with col22:
        prediction_plot(model,prediction)

    with col23:
        prediction_plot(model,prediction)
     
    with col24:
        prediction_plot(model,prediction)

    with col25:
        prediction_plot(model,prediction)

    
    col26, col27, col28,col29,col30 = st.beta_columns(5)
    
    with col26:
        prediction_plot(model,prediction)
     
    with col27:
        prediction_plot(model,prediction)

    with col28:
        prediction_plot(model,prediction)
     
    with col29:
        prediction_plot(model,prediction)

    with col30:
        prediction_plot(model,prediction)

    st.subheader("Model Qualitative Data")

    col31,col32= st.beta_columns(2)
    with col31:
        st.markdown("Confusion Metrics")
        disp = metrics.plot_confusion_matrix(model["model"], model["X_test"],model["Y_actual"])
        st.write(disp.confusion_matrix)

    with col32:
        classification_report=metrics.classification_report(model["Y_actual"],prediction,output_dict=True)
        st.markdown("Classification Report")
        st.write(pd.DataFrame({
            "Class Name" :[0,1,2,3,4,5,6,7,8,9] ,
            "Precision" : [classification_report["0"]["precision"], classification_report["1"]["precision"], 
                          classification_report["2"]["precision"], classification_report["3"]["precision"],
                          classification_report["4"]["precision"], classification_report["5"]["precision"],
                          classification_report["6"]["precision"],  classification_report["7"]["precision"], 
                          classification_report["8"]["precision"], classification_report["9"]["precision"]],
            "Recall" : [classification_report["0"]["recall"], classification_report["1"]["recall"], 
                          classification_report["2"]["recall"], classification_report["3"]["recall"],
                          classification_report["4"]["recall"], classification_report["5"]["recall"],
                          classification_report["6"]["recall"],  classification_report["7"]["recall"], 
                          classification_report["8"]["recall"], classification_report["9"]["recall"]],
             "F1-Score" : [classification_report["0"]["f1-score"], classification_report["1"]["f1-score"], 
                          classification_report["2"]["f1-score"], classification_report["3"]["recall"],
                          classification_report["4"]["f1-score"], classification_report["5"]["f1-score"],
                          classification_report["6"]["f1-score"],  classification_report["7"]["f1-score"], 
                          classification_report["8"]["f1-score"], classification_report["9"]["f1-score"]],
             "Support" : [classification_report["0"]["support"], classification_report["1"]["support"], 
                          classification_report["2"]["support"], classification_report["3"]["support"],
                          classification_report["4"]["support"], classification_report["5"]["support"],
                          classification_report["6"]["support"],  classification_report["7"]["support"], 
                          classification_report["8"]["support"], classification_report["9"]["support"]],
            
            }))

    
    col33,col34= st.beta_columns(2)
 
    with col33:
        st.markdown("Accuracy and Macro Average")
        st.write(pd.DataFrame({
            "Accuracy" :[classification_report["accuracy"]] ,
            "Precision" :[classification_report["macro avg"]["precision"]],
            "Recall"    :[classification_report["macro avg"]["recall"]],
            "F1-Score" : [classification_report["macro avg"]["f1-score"]],
            "Support" : [classification_report["macro avg"]["support"]],
             }))

    with col34:
        st.markdown("Weighted average")
        st.write(pd.DataFrame({
            "Precision" :[classification_report["weighted avg"]["precision"]],
            "Recall"    :[classification_report["weighted avg"]["recall"]],
            "F1-Score" : [classification_report["weighted avg"]["f1-score"]],
            "Support" : [classification_report["weighted avg"]["support"]],
             }))
 

with st.beta_container():
    components.html(
     """
     <div style="position: fixed;
   left: 0;
   bottom: 0;
   width: 100%;
   background-color: black;
   color: white;
   text-align: center;">
  <p>Ravi Shekhar Tiwari</p>
</div>
     """,height=140,)

