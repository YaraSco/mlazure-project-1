# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**
This project uses the tabular dataset in the URL "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv". This dataset has 19 attributes for training and the attribute 'y' for predictions. The attribute 'y' has two classes 'no' and 'yes'. It determines if a potentiel client would accept to make a deposit at the bank or decline the offer.

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**
The model used by hyperdrive and provided by Scikit-learn, LogisticRegression, had an accuracy of 0.9095. However, the model obtained by an AutoMl run had a better accuracy with 0.9181. So, the best performing model was Voting Ensemble of the Automl run.

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
<img src="./images/pipeline architecture.PNG">

The diagram above represents the pipeline architecture of our project. We begin with creating the dataset from the provided link. After that, we clean our data using the "train.py" function "clean_data(data)". The function clean and one hot encode data, so the attribute 'y' is transformed from categorical ('no' and 'yes') to a numeric attribute (0 and 1). We split the data in the python script "train.py" for the hyperdrive usage and in the notebook for the AutoML run. We add another step for the AutoMLto register the data in a default datastore.

The steps for hyperparameter tuning :
Firstly, we define the search space for the model's hyperparameters "C" and "max_iter". We will sample the hyperparameters randomly.
"C" is continuous, so we choose randomly and uniformly from a range of [0.05, 10]. 
"max_iter" is descrete, so we choose randomly from these numbers 100, 300, 500, 700 and 1000.
Secondly, we specify the "Accuracy", defined in "train.py", as the primary metric to maximize.
Thirdly, we define an early termination policy, Bandit Policy.
Finally, we configure settings for resource allocation. The maximum number of training runs is "max_total_runs=24". We run concurrently 4 runs as our compute target is capable of. 

The classification algorithm is the Logistic Regression provided by  Scikit-learn. Indeed, we want to predict the attribute 'y' with classification, because as we explained, 'y' has two classes '0' and '1'. In the "train.py" script, we parse two hyperparameters "C" and "max_iter". 
 - "C" represents the inverse of regularization strength. We discover the range of (0.05, 10). We choose until 10, because smaller values cause stronger regularization. Th default is 1.0
 - "max_iter" represents the maximum number of iterations to converge. The default is 100. We choose higher values in our hyperparameter tuning 

**What are the benefits of the parameter sampler you chose?**
Azure Machine Learning supports 3 sampling methods:
  Grid sampling. It only supports choice. So, we can not use it for "C", which uses unifom function.
  Bayesian sampling. It chooses based on the previous samples, and it is much costly in terms of the budget.
  Random sampling. It supports the functions we need "choice" and "uniform". Also, it is not as costly as the Bayesian sampling. Moreover, it supports early termination of low-performance runs.
So, we chose random sampling as the parameter sampler. 

**What are the benefits of the early stopping policy you chose?**
We chose "BanditPolicy(slack_factor=0.1, evaluation_interval=1, delay_evaluation=3)" as the early stopping policy. The benefits are that this policy is based on the slack factor that terminates any run whose best metric is less than it. Also, we delay the first policy evaluation to the third interval. 

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
