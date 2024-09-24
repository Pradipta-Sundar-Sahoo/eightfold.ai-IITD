# INNOV8: The Space Saga (IITD & Eightfold.ai)
[GITHUB LINK](https://github.com/Pradipta-Sundar-Sahoo/eightfold.ai-IITD)
## PARTS
1. [PART 1](#part-1)
2. [PART 2](#part-2)
_______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
# PART 1
# Decoding and Classifying Alien Communications
## Overview
**INNOV8: The Space Saga** is a collaborative project between ARIES IITD and Eightfold.ai, focusing on decoding and classifying alien communications. The objective is to utilize machine learning techniques to identify and classify alien messages from an intercepted dataset.

The project involves several key steps including Exploratory Data Analysis (EDA), data preprocessing, feature extraction, model training, and performance evaluation.

## Table of Contents
1. [Project Description](#project-description)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Model Performance](#model-performance)
5. [Overfitting Analysis](#overfitting-analysis)
6. [Final Model Selection](#final-model-selection)

## Project Description
The dataset consists of intercepted alien transmissions with the following structure:

- **Message**: The alien communication in text form.
- **Species**: The species that sent the message.
- **Additional Features**: Characteristics such as number of fingers, presence of a tail, etc.

### Goal:
Develop machine learning models to classify the alien species based on these features.

### Workflow:
1. **Exploratory Data Analysis (EDA)**: Analyze the dataset to understand the distribution and relationships of features.
2. **Data Preprocessing**: Clean and transform the data for training the models.
3. **Modeling**: Test and evaluate various machine learning models, including SVM, XGBoost, Logistic Regression, GRU, and Random Forest.
4. **Overfitting Analysis**: Ensure models generalize well to unseen data.
5. **Final Training**: Fine-tune the best-performing model.

## Installation
To run this project, ensure the following dependencies are installed:

```bash
pip install -r requirements.txt
```

### Key Libraries:
- `pandas`
- `scikit-learn`
- `seaborn`
- `matplotlib`
- `transformers`
- `torch`

## Usage
1. **Data Loading**: The dataset (`data.csv`) is loaded, and basic exploratory analysis is performed.
2. **Preprocessing**: The dataset is cleaned and transformed, and relevant features are extracted.
3. **Model Training**: Several models are trained and evaluated based on their performance metrics.
4. **Visualization**: Data and model performance are visualized using `seaborn` and `matplotlib`.

To run the notebook:

```bash
jupyter notebook solution.ipynb
```
## Model Performance
These models are used:

![image](https://github.com/user-attachments/assets/1814ea05-f99c-432b-92d5-f2226ed4234c)


## Overfitting Analysis
![image](https://github.com/user-attachments/assets/f24bb8a2-68ef-40b4-ad57-5609c006e330)
![image](https://github.com/user-attachments/assets/1bfe7d09-6838-455f-b3cf-a6f8d468b0c3)
![image](https://github.com/user-attachments/assets/cabb509e-808a-4a41-980a-ba4a471a8c6a)
![image](https://github.com/user-attachments/assets/a8b3aadf-7b91-45e8-8dcf-fa66a63caec2)



## Final Model Selection
After analyzing the performance of various models, the **SVM with linear kernel** was selected for final training due to its balanced performance on both training and test data, as well as its ability to generalize well without overfitting.

______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
# PART 2
# Predicting Troop Betrayal in the War Against the Phrygians

## Overview
This project focuses on predicting the likelihood of soldiers betraying their clan during the war against the Phrygians. The system uses machine learning techniques to evaluate various risk factors such as financial status, disciplinary record, proximity to enemy territory, and social bonds. The aim is to flag high-risk soldiers and improve decision-making to prevent betrayal.

## Table of Contents
1. [Problem Description](#problem-description)
2. [Key Features](#key-features)
3. [Mathematical Model](#mathematical-model)
4. [System Workflow](#system-workflow)
5. [Model Performance](#model-performance)
6. [Installation](#installation)
7. [Dataset Creation](#dataset-creation)

## Problem Description
As the leader of the Xernian army, the challenge is to identify potential traitors based on various factors. The system aims to develop a decision-making model that predicts the likelihood of betrayal based on historical and behavioral data.

## Key Features
- **Income Percentile**: Soldiers' financial status relative to their peers.
- **Disciplinary Record**: Number and severity of infractions in their record.
- **Proximity to Enemy Territory**: Distance to the Phrygian border.
- **Social Bonds**: Number of comrades a soldier interacts with.
- **Family History of Betrayal**: Whether a soldier's family has a history of defection.

## Mathematical Model (Logic for dataset creation)
The risk score \( R_i \) for each soldier is calculated as a weighted sum of these features:

![image](https://github.com/user-attachments/assets/6f8816db-eba0-417f-947f-2f0ed098d6b2)


Where \( wj \) represents the weight for each feature, and \( xij \) is the normalized value of the feature for soldier \( i \). Soldiers with \( Ri > T \) (a predefined threshold) are flagged as high-risk.

## System Workflow
1. **Data Collection**: Gather data on income, disciplinary records, proximity to the enemy, social bonds, and family history.
2. **Preprocessing**: Normalize data and convert qualitative values (like family history) into numerical representations.
3. **Risk Score Calculation**: Calculate risk scores for each soldier using the weighted sum formula.
4. **Decision-Making**: Soldiers with risk scores above the threshold are flagged for potential betrayal.
5. **Adaptation**: The system uses machine learning models (e.g., Random Forest, Logistic Regression) to retrain and adjust the weights dynamically as new data is gathered.

## Model Performance
- **Random Forest (without SMOTE)**: Achieved **94% accuracy** on an imbalanced dataset.
- **Random Forest (with SMOTE)**: Achieved **89.5% accuracy** after balancing the dataset.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/troop-betrayal-prediction.git
## Dataset Creation
![image](https://github.com/user-attachments/assets/8596cf84-8c71-4a07-933a-f7ec4d0fe79e)

