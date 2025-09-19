# Approximated BCNN Active Learning and Training on AL data - Site B 2020 
This directory contains the Jupyter Notebooks for the BAL of Approximated Bayesian CNN (A-BCNN) and then the training of A-BCNN on the whole AL data collected by T-BCNN, A-BCNN and deterministic CNN to compare between all three models.

## Requirements

* Pytorch 3.8.12 
* PyTorch 1.11.0

## File Structure

* <b>BAL using Approximated BCNN.ipynb</b>: Contains the implementation of BAL techniques (Predictive Entropy, BALD, Variation Ratio, BvSB) on the A-BCNN model.
* <b>BAL_history_approxBCNN.xlsx</b>: Contains the history of the four BAL techniques on A-BCNN model. An excel report.
* <b>Approx BCNN Training - AL Data SiteB2020.ipynb</b>: Contains the implementation of training and evaluation of the A-BCNN model on the Active Learning data collected by the AL of T-BCNN, A-BCNN and Deterministic CNN.
* <b>approx_BCNN_training_evaluation.xlsx</b>: Contains the history and evaluation results of all AL data on A-BCNN model. An excel report.
* <b>approx_all_final_results.xlsx</b>: Contains the final evaluation of all AL data on A-BCNN model for comparison of which samples performed better on this model. An excel report.

<hr>
