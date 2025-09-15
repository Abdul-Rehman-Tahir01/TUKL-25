# Active Learning using Approximate Bayesian Script
This script provides an implementation for the Active Learning using Approximate Bayesian (MC Dropouts).
<ul>
    <li>Bayesian Active Learning by Disagreement (BALD)</li>
    <li>Best Vs Second Best (BVSB)</li>
</ul>
You can change the parameter values and experiment with the differnt configurations. 

## Requirements

* Pytorch 3.8.12 
* PyTorch 1.11.0

## File Structure

* <b>approx_bal_script.py</b>: The actual approximate bayesian active learning script.
* <b>dataset.py</b>: Contains the dataset class.
* <b>model.py</b>: Contains the model architecture.
* <b>acquisition_functions.py</b>: Contains acquisition functions.
* <b>evaluation.py</b>: Contains the evaluation function.
<br><br>
* <b>{strategy_name} directory</b>: When you run the script, a directory named on the strategy you ran will get created which have the data, model checkpoints, and results.
* <b>results/excel_history_{strategy_name}.xlsx</b>: Excel file for result logging. Created when running the script.
* <b>results/history_{strategy_name}.json</b>: JSON file for history results.
* <b>data/approx_bcnn_data_{strategy_name}.npz</b>: A npz file which have the data samples selected by that active learning strategy.
* <b>checkpoints/checkpoint_{strategy_name}_round_{round_num}.pth</b>: Model checkpoint. Created when running the script. If due to any reason the script stops, it is resumed automatically by loading the checkpoint.
* <b>args_info.txt</b>: Text file for information of all the arguments that can be used.

You can change any of the four utility files (dataset.py, model.py, query_strategies.py, evalutation.py) according to requirements.

## Results Logging and Checkpoints

-> The results are being stored in the JSON file per Active Learning round in a txt file in python dictionary format and has following structure:

```
{
    "rounds": [1, 2, 3],
    "num_labeled": [50, 100, 150],
    "train_accuracy": [_, _, _],
    "train_loss": [_, _, _],
    "val_accuracy": [_, _, _],    
    "val_loss": [_, _, _],
    "avg_f1": [_, _, _],      
    "per_class_f1": [
        [_, _, _],          # Per class f1
        [_, _, _],          for each round
        [_, _, _],
    ],
    "kappa": [_, _, _],
    "confusion_matrices": [
        [[_, _, _], [_, _, _], [_, _, _]],      # CM for each round
        [[_, _, _], [_, _, _], [_, _, _]],
        [[_, _, _], [_, _, _], [_, _, _]],
    ]
}
```

Loading and reading the results:

```
import json

with open("history_{strategy}.txt", "r") as f:
    history = json.load(f)

# Now `history` is a Python dictionary
print(history["rounds"])
print(history["val_accuracy"])
print(history['confusion_matrices'])

```
<br>
-> The checkpoints are also being created at each AL round, stored in pth format and has following structure:

```
{
    'round': round_num,
    'backbone_state_dict': backbone.state_dict(),
    'fc_state_dict': fc.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'labeled_pool': list(labeled_pool),
    'unlabeled_pool': list(unlabeled_pool),
    'args': vars(args),
}
```

Loading and reading the checkpoints:

```
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

# Restore states
backbone.load_state_dict(checkpoint['backbone_state_dict'])
fc.load_state_dict(checkpoint['fc_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

labeled_pool = set(checkpoint['labeled_pool'])
unlabeled_pool = set(checkpoint['unlabeled_pool'])
```

## Usage 


```shell
python approx_bal_script.py
```

Example: With strategy selection, epoch, stochastic passes, learning rate

```shell
python bal_script.py --strategy bald --epochs 10 --stochastic_passes 20 --lr 1e-6
```

Example: With data directory, backbone and fc path
```shell
python bal_script.py --input_dir /abc/123 --backbone_path /abd/123/backbone --fc_path /abc/123/fc
```

<hr>
