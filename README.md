# MHR_classification

Machine Learning Model Deployment on AWS

## **1. Project Overview**

This project involves building and deploying a machine learning model on AWS. The system includes data preprocessing, model training, and inference served through a Flask API. The goal is to create an efficient and scalable ML pipeline for real-time predictions.

## **2. Technology Stack**

* **Programming Language:** Python

* **Libraries:** Pandas, NumPy, Scikit-learn, Flask

* **Model Deployment:** Flask API

* **Cloud Platform:** AWS (EC2, S3, Lambda)

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

* Hosted the Flask API on an AWS EC2 instance.

* Configured AWS S3 for storing model artifacts.

* Set up an Nginx reverse proxy for handling requests.

## **4. Installation & Setup**

**4.1. Local Setup**

Clone the repository:

```
git clone https://github.com/your-repo/ml-deployment.git
cd ml-deployment
```

Create a virtual environment and install dependencies:

```
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

Run the Flask API locally:

```
python app.py
```

**4.2. AWS Deployment**

Launch an EC2 Instance

Choose an Ubuntu AMI and set up security groups.

SSH into the instance:

```
ssh -i your-key.pem ubuntu@your-ec2-ip
```

Install dependencies and start the Flask API:

```
sudo apt update && sudo apt install python3-pip nginx -y
pip3 install -r requirements.txt
python3 app.py
```

Configure Nginx for Reverse Proxy

Edit Nginx configuration:

```
sudo nano /etc/nginx/sites-available/default
```

Add the following:

```
server {
    listen 80;
    server_name your-ec2-ip;
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Restart Nginx:

```
sudo systemctl restart nginx
```

Testing the Deployment

```
Access the API at http://your-ec2-ip/predict.
```

Send a request using:

```
curl -X POST http://your-ec2-ip/predict -H "Content-Type: application/json" -d '{"input": [your_data]}'
```

## **5. Future Enhancements**

* Implement model versioning with AWS Lambda.

* Integrate a CI/CD pipeline for automated deployments.

* Optimize API response time with AWS SageMaker.

## **6. Conclusion**

This project demonstrates an end-to-end machine learning workflow from data preprocessing to cloud deployment. The integration with AWS ensures scalability and accessibility for real-world applications.

📌 Author: [Your Name]

🔗 GitHub Repository: [Repo Link]
