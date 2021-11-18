LOG_DATA_PKL    =  "data.pkl"
LOG_MODEL_PKL   =  "model.pkl"

#-------------------------------------------------------------

import os 
import sklearn
import pickle
import yaml

import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient

#-------------------------------------------------------------


class JobPrediction:
    """Production Class for predicting the probability of a job from skills"""
    # Class attributes 
    path_clusters_config = None
    skills_clusters_df   = None
    
    artifacts_path = None
    
    model        = None
    all_features = None 
    all_jobs     = None 
    
    # Constructor
    def __init__(self, artifacts_path, clusters_yaml_path):
        # Store variables
        self.artifacts_path = artifacts_path
        
        # Retrieve model and features
        mlflow_objs = self.load_mlflow_objs(artifacts_path)
        self.model        = mlflow_objs[0]
        self.all_features = mlflow_objs[1]
        self.all_jobs     = mlflow_objs[2]
        
        # Load clusters config 
        self.path_clusters_config = clusters_yaml_path
        self.skills_clusters_df = self.load_clusters_config(clusters_yaml_path)
        
    # -------------------------------------------
    
    # Constructor helper functions 
    
    def load_mlflow_objs(self, artifacts_path):
        """Load objects from the MLflow run"""

        
        # Load data pkl
        data_path  = os.path.join(artifacts_path, LOG_DATA_PKL)
        with open(data_path, 'rb') as handle:
            data_pkl = pickle.load(handle)
            
        # Load model pkl
        model_path = os.path.join(artifacts_path, LOG_MODEL_PKL)
        with open(model_path, 'rb') as handle:
            model_pkl = pickle.load(handle)

        # Return model and data labels 
        return model_pkl["model_object"], data_pkl["features_names"], data_pkl["targets_names"]
    
    
    def load_clusters_config(self, path_clusters_config):
        """Load skills clusters developed in 03_feature_engineering.ipynb"""
        
        # Read YAML
        with open(path_clusters_config, "r") as stream:
            clusters_config = yaml.safe_load(stream)
            
        # Format into dataframe
        clusters_df = [(cluster_name, cluster_skill)
                       for cluster_name, cluster_skills in clusters_config.items()
                       for cluster_skill in cluster_skills]
        
        clusters_df = pd.DataFrame(clusters_df, 
                                   columns=["cluster_name", "skill"])
        return clusters_df

    
    # ========================================================
    # **************    Prediction Functions    **************  
    # ========================================================
    
    def create_features_array(self, available_skills):
        """Create the features array from a list of the available skills"""
        
        # Method's helper functions 
        def create_clusters_features(self, available_skills):
            sample_clusters = self.skills_clusters_df.copy()
            sample_clusters["available_skills"] = sample_clusters["skill"].isin(available_skills)
            cluster_features = sample_clusters.groupby("cluster_name")["available_skills"].sum()
            return cluster_features
            
        def create_skills_features(self, available_skills, exclude_features):
            all_features = pd.Series(self.all_features.copy())
            skills_names = all_features[~all_features.isin(exclude_features)]
            ohe_skills = pd.Series(skills_names.isin(available_skills).astype(int).tolist(), 
                                   index=skills_names)
            return ohe_skills
        
        # -------------------------
        
        # Method's main
        clusters_features = create_clusters_features(self, available_skills)
        skills_features   = create_skills_features(self, available_skills, 
                                                   exclude_features=clusters_features.index)
        # ... Combine features and sort 
        features = pd.concat([skills_features, clusters_features])
        features = features[self.all_features]
        return features.values 
    
    
    def predict_jobs_probabilities(self, available_skills):
        '''Returns probabilities of the different jobs according to the skills'''
        # Create features array 
        features_array = self.create_features_array(available_skills)
        
        # Predict and format
        predictions = self.model.predict_proba([features_array])
        predictions = [prob[0][1] for prob in predictions] # Keep positive probs 
        predictions = pd.Series(predictions, index=self.all_jobs)
        
        return predictions

    
    # ==============================================================