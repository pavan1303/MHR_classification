# MHR_classification

Machine Learning Model Deployment on AWS

## **1. Project Overview**

This project involves building and deploying a machine learning model on AWS. The system includes data preprocessing, model training, and inference served through a Flask API. The goal is to create an efficient and scalable ML pipeline for real-time predictions.

## **2. Technology Stack**

* **Programming Language:** Python

* **Libraries:** Pandas, NumPy, Scikit-learn, Flask

* **Model Deployment:** Flask API

* **Cloud Platform:** AWS (Elastic Bean Stalk)

## **3. Project Workflow**

 **3.1 Exploratory Data Analysis (EDA)**

* Identified missing values and outliers.

* Performed feature engineering and scaling.

**3.2 Preprocessing and Model Training**

* Cleaned and transformed raw data.

* Trained a machine learning model using Scikit-learn.

* Saved the trained model for deployment.

**3.3 Building the Flask API**

* Created an API with Flask to serve predictions.

* Implemented routes for model inference.

**3.4 Deployment on AWS**

* Deployed the Flask API using AWS Elastic Beanstalk.

* Automated deployments using AWS CodePipeline.

## **4. Installation & Setup**

**4.1. Local Setup**

Clone the repository:

```
git clone git@github.com:pavan1303/MHR_classification.git
```

Create a virtual environment and install dependencies:

```
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install .
```

Run the Flask API locally:

```
python application.py
```

# **Demo**
## **Before Prediction**

![Screenshot (680)](https://github.com/user-attachments/assets/6b55a548-36ce-43db-96d1-2d32588a289f)




--------------------------------------------------------------------------------------------------------------
## **After Prediction**
![Screenshot (686)](https://github.com/user-attachments/assets/d13089ce-dbde-4896-bf6f-3b585d57e0a2)


**4.2. AWS Deployment**

* Initialize Elastic Beanstalk Application

* Create and Deploy the Environment

* Deploy Updates

## **5. Future Enhancements**

* Integrate a CI/CD pipeline for automated deployments.

## **6. Conclusion**

This project demonstrates an end-to-end machine learning workflow from data preprocessing to cloud deployment. The integration with AWS ensures scalability and accessibility for real-world applications.

📌 Author: Ella Pavan Kumar

🔗 GitHub Repository: git@github.com:pavan1303/MHR_classification.git
