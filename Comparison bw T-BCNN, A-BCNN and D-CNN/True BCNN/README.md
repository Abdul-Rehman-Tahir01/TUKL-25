# True BCNN Active Learning and Training on AL data - Site B 2020 
This directory contains the Jupyter Notebooks for the BAL of True Bayesian CNN (T-BCNN) and then the training of T-BCNN on the whole AL data on Site B 2020 collected by T-BCNN, A-BCNN and Deterministic CNN to compare between all three models.

## Requirements

* Pytorch 3.8.12 
* PyTorch 1.11.0

## File Structure

* <b>BAL using True BCNN.ipynb</b>: Contains the implementation of BAL techniques (Predictive Entropy, BALD, Variation Ratio, BvSB) on the T-BCNN model.
* <b>BAL_history_trueBCNN.xlsx</b>: Contains the history of the four BAL techniques on T-BCNN model. An excel report.
* <b>True BCNN Training - AL Data SiteB2020.ipynb</b>: Contains the implementation of training and evaluation of the T-BCNN model on the Active Learning data (Site B 2020) collected by the AL of T-BCNN, A-BCNN and Deterministic CNN.
* <b>true_BCNN_training_evaluation.xlsx</b>: Contains the history and evaluation results of all AL data on T-BCNN model. An excel report.
* <b>true_all_final_results.xlsx</b>: Contains the final evaluation of all AL data on T-BCNN model for comparison of which samples performed better on this model. An excel report.

<hr>
