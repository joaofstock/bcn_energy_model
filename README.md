# bcn_energy_model
Energy model predictor of Barcelona Neighborhoods

This repository contais: 
app.py - the flask to run the model and load the prediction in the html page
templates/index.html - the html code to load the page
static/styles.css - the webpage styles
Dockerfile - to build the image in the EC2
model.ipynb - training and test data and source of pkl files, ran by the flask (pkls are already attached, but this file updates them)
synthetic_df.ipynb - syntetic data generator for future houses age, neighborhood population and weather
requirements.txt - py packages required
