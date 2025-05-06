# MachineLearning

Homework 8 - Regression with neural networks.  

  Using a neural net, we are able to accuratly follow the curve of any function with enough epochs. The Jupyter notebook attached shows the function creation as well as the curves being fit by our neural network. The functions are shown on the jupyter notebook, where we added in a small amount of noise to our function to give the extra data points.   
I used the torch import with F.tanh as the activation function and ran for 1000 epochs. Below is a graph of a complex function with the red line being our Neural net approximating the function very well.   

![image](https://github.com/user-attachments/assets/b7ef3669-24b0-4a96-997f-3b2a7bf4c6a5)

  
Homework 7 - Clustering taxi locations.   

  Based on the given latitude and longitude of taxi pickup and dropoff locations, we are able to apply kmeans and clustering to display a map of the main locations where the pickups and dropoffs are located. This would be useful when trying to decide locations to place taxi distribution centers, and main locations of which people need to be picked up or dropped off by a taxi.  

Using Kmeans in this scope allows us to see the main areas where the pickup/dropoff locations are the most dense, and how with this data we could determine where we want to place hubs for the cab company. They could make interpretations on where people are picked up based on time of day, and then use these interpretations to make informed decisions on where they want to place their hubs.  
  
  https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data?select=yellow_tripdata_2015-01.csv
  This is a link to the kaggle dataset used for this project, with one month of data.  

  Here is an image of what the map looks like after using clustering to determine main central locations of the pickup spots
  ![image](https://github.com/user-attachments/assets/f54dcc43-9b8f-4e51-9b6c-31c368d5d5c3)

  
Homework 6 - 
We are trying to determine what features are most important when classifying what type of credit the given individual has, and then use a random forrest classifier to try and classify the test data based on those parameters. We had to clean the data in order to only have the specific variables we needed. 
Below is a link to the dataset on kaggle for reference. 
https://www.kaggle.com/datasets/parisrohan/credit-score-classification

Below is the importance of each feature ranked using the dataframe function. This gives us insight into what features have the largest impact on our classification
![image](https://github.com/user-attachments/assets/919c1817-ebab-44d3-9d7d-480dbcde625a)

After we ran this with our data, below is the confusion matrix of our classification on the test data. This shows us how well we did in our calssification on each different category and what the overall accuracy of our method is. 
![image](https://github.com/user-attachments/assets/f8072dd5-7c47-4af7-8737-1c7951c8b7f9)  

There are many important features that go into credit, and the first bar graph clearly displays our important features and the confusion matrix expresses how well we do on each of our different credit types. Overall this method using random forrest classifier did very well with a high accuracy score.   


Homework 4 - 
We are attempting to determine the weather type based on a given dataset that has weather information using SVR and RBF. This allows us to determine given weather data what type of weather the data is currently in. The function does pretty well at determining what type it is, with 0.94 accuracy on each of our different types. 

Here is the link to the kaggle dataset. 
https://www.kaggle.com/datasets/nikhil7280/weather-type-classification

Below is the confusion matrix based on our calculations 
![image](https://github.com/user-attachments/assets/3957d2fc-8dd4-430a-9c0b-1ed56d7c0af5)

This shows how SVR with RBF does a very good job at classifying what weather type it is, and that this model has good accuracy with each variable having a high percentage of correct classifications. 
