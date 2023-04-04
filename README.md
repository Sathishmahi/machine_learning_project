# End to End ML project

## Problem Statement 

To predict the median house value of California State(U.S)

## about data
in this data collected information on the variables using all the block groups in California from the 1990 Cens us.

## Total no of Rows
20640 

## Total no of Columns
10

## Problem Type

Regression

## dataset variables(columns)

longitude               

latitude                

housing_median_age      

total_rooms             

total_bedrooms          

population              

households              

median_income           

ocean_proximity

median_house_value 


## Traget Column

median_house_value 


## Training Pipe Line

![training_pipeline](https://user-images.githubusercontent.com/88724458/229689630-995e6372-245e-442a-a616-ca2d42f29d76.png)


## Prediction Pipe Line

![prediction_pipeline](https://user-images.githubusercontent.com/88724458/229689695-ffa9dd02-56fd-4153-85f8-472dfa54cbaa.png)



## Tech Stack

**Language:** Python

**Web Framework:** Python Flask

**ML Libraries to Used:** Pandas , Numpy , Scikit-learn

**ML Alogorithms to Used:** 
LightGBM , XGBoost , ADABoost , Ridge , Lasso ,  Elasticnet , DecisionTree , RandomForest , GradientBoost , KMeans(cluster) , KNN(to handle the outlier)


## Run Locally


Clone the project

```bash
git clone https://github.com/Sathishmahi/machine_learning_project.git
```

create conda env

```bash
conda create -p venv python=3.7 -y
```

activate conda env

```bash
conda activate venv/
```

install dependencies

```bash
pip install -r requirements.txt
```

run test.py to train models

```bash
python test.py
```

run server

```bash
python app.py
```

## any doubt reach me
### sathishofficial456@gmail.com
