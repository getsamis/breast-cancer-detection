# Breast Cancer Detection

A machine learning project for classifying breast cancer tumors as **Malignant or Benign** using `scikit-learn`, orchestrated and scaled in the cloud using **AWS SageMaker**.

## 🎯 Project Motivation & Learning Goals
This project was built as a hands-on exploration to bridge the gap between local machine learning experimentation and cloud production. By migrating the workflow to **AWS SageMaker** and structured scripts, the core goals were to:
- Gain practical, real-world experience with cloud orchestration and remote training jobs.
- Learn how to structure ML code for production (`train.py` + orchestration notebooks).
- Deepen understanding of AWS infrastructure and cloud-based machine learning pipelines.

## 🚀 Project Architecture & Workflow
- **`run_pipeline.ipynb`**: End-to-end SageMaker orchestration notebook handling cloud training jobs and workflow execution.
- **`train.py`**: Standalone training script executed securely within AWS SageMaker infrastructure.
- **`Logistic_Regression.ipynb`**: Initial version comprised of exploratory data analysis, cleaning, processing visualization notebook, training, prediction and evaluation.

## 🔑 Key Features & Pipeline Steps
- **Data Analysis & Cleaning:** Handled missing values and structured data integrity.
- **Data Processing:** Standardized feature scaling for optimal model convergence.
- **Data Visualization:** Generated insights into feature correlations.
- **Cloud Execution:** Leveraged Amazon SageMaker and `boto3` to run reproducible training jobs in isolated cloud environments.
