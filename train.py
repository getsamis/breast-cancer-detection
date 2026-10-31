#argument parsing 
import argparse 
import joblib
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #Hyperparameters for Logistic Regression
    parser.add_argument("--C", type = float, default=1.0
    parser.add_argument("-max_iter", type=int, default=1000

    #Sagemaker specific environment paths
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type =str, default=os.environ.get("SM_CHANNEL_TRAIN"))

    args, _ = parser.parse_known_args()

    #Load the data
    print(f"Loading training data....")
    train_file = os.path.join(args.train, "breast_cancer_data.csv")
    df = pd.read_csv("train_file") 

    #Data Preprocessing 
    #--Drop id column 
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    #Map diagnosis column
    if "diagnosis" in df.columns: 
       
       #M: Malignant, B: Benign
        df["diagnosis"] = df["dignosis"].map({"M": 1, "B":0})
        
       #Target(y) and Features (X) 
        X = df["diagnosis"]
        y = df.drop(columns=["diagnosis"]
     else:
        #Fallbak if diagnosis is the first column
        y = df.iloc[:,0]
        X = df.iloc[:,1:]
        print(f"Dataset shape: {X.shape}")


    #Model Training
    print("Training Logistic Regression with C={args.C}, max_iter={args.max_iter}....")
    model = LogisticRegression(C=args.C, max_iter=args.max_iter, random_state=42)
    model.fit(X,y)

    #Serialisation and Saving
    #Serialise
    model_path = os.path.join(args.model_dir, "model.joblib")
    #Save
    joblib.dump(model, model_path)
    print(f"Model succesfully saved to {model_path")
