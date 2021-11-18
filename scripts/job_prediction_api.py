ARTIFACTS_PATH = 'C:/Users/M/Documents/data_science_project/notebooks/mlruns/1/67403f4417c34d82b102947d9e1aaa9d/artifacts/'
CLUSTERS_YAML_PATH = "C:/Users/M/Documents/data_science_project/data/processed/features_skills_clusters_description.yaml"

#------------------------------------------

import JobPrediction
from JobPrediction import JobPrediction

import pandas as pd
from flask import Flask, request, jsonify

#------------------------------------------

# Initiate API and JobPrediction object
app = Flask(__name__)
job_model = JobPrediction(artifacts_path = ARTIFACTS_PATH, 
                          clusters_yaml_path=CLUSTERS_YAML_PATH)


# Create prediction endpoint 
@app.route('/predict_jobs_probs', methods=['POST'])
def predict_jobs_probs():
    available_skills = request.get_json()
    predictions = job_model.predict_jobs_probabilities(available_skills).to_dict()    
    return jsonify(predictions)


if __name__ == '__main__':
    app.run()


