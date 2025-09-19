# Comparison between True BCNN, Approximated BCNN and Deterministic CNN on Site B 2020 data 
This directory contains the implementation of the comparison between the three models (T-BCNN, A-BCNN and D-CNN) and the data samples collected via active learning using these three models. The models weights used are those weights which we get by training them on Site A 2019 (25%) data. The data used for Active Learning is of Site B 2020.

## Requirements

* Pytorch 3.8.12 
* PyTorch 1.11.0

## File Structure

* <b>Approximated BCNN Directory</b>: Contains all the implementation done to first train the model on AL data (using the techniques listed inside the readme of this directory) to get AL samples by A-BCNN model and then the training of this model on the samples selected by all three models.

* <b>True BCNN Directoryx</b>: Contains all the implementation done to first train the model on AL data (using the techniques listed inside the readme of this directory) to get AL samples by T-BCNN model and then the training of this model on the samples selected by all three models.

* <b>Deterministic CNN Directory</b>: Contains all the implementation done to first train the model on AL data (using the techniques listed inside the readme of this directory) to get AL samples by D-CNN model and then the training of this model on the samples selected by all three models.

* <b>complete_comparison.xlsx</b> Contains the complete comparison between all three models and all 13 data sampling strategies (AvgF1 score is listed).

<hr>