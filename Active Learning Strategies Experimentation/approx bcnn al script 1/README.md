# Bayesian Active Learning Script
This script provides an implementation for the three Bayesian Active Learning techniques.
<ul>
    <li>Predictve Entropy</li>
    <li>Bayesian Active Learning by Disagreement (BALD)</li>
    <li>Variation Ratio</li>
</ul>
You can change the parameter values and experiment with the differnt configurations. 

## Requirements

* Pytorch 3.8.12 
* PyTorch 1.11.0

## File Structure

* <b>bal_script.py</b>: The actual bayesian active learning script.
* <b>dataset.py</b>: Contains the dataset class.
* <b>model.py</b>: Contains the model architecture.
* <b>query_strategies.py</b>: Contains query sampling strategies functions.
* <b>evaluation.py</b>: Contains the evaluation function.
* <b>history_{strategy}.txt</b>: Text file for result logging. Created when running the script.
* <b>checkpoint_{strategy}\_round_{round_num}.pth</b>: Model checkpoint. Created when running the script.
* <b>args_info.txt</b>: Text file for information of all the arguments that can be used.

You can change any of the four utility files (dataset.py, model.py, query_strategies.py, evalutation.py) according to requirements.

## Results Logging and Checkpoints

-> The results are being stored per Active Learning round in a txt file in python dictionary format and has following structure:

```
{
    "rounds": [1, 2, 3],
    "num_labeled": [50, 100, 150],
    "train_accuracy": [
        [_, _, _, _],      # Per epoch accuracy
        [_, _, _, _],      for each round
        [_, _, _, _],
    ],
    "train_loss": [
        [_, _, _, _],      # Per epoch loss
        [_, _, _, _],      for each round
        [_, _, _, _],
    ],
    "val_accuracy": [_, _, _],    # Val accuracy for each round
    "val_avg_f1": [_, _, _],      # Val avg f1 for each round
    "val_f1s": [
        [_, _, _],          # Per class f1
        [_, _, _],          for each round
        [_, _, _],
    ],
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
checkpoint = torch.load('checkpoint_{strategy}_round_{round_num}.pth')

# Restore round number and history
round_num = checkpoint['round']
labeled_pool = set(checkpoint['labeled_pool'])
unlabeled_pool = set(checkpoint['unlabeled_pool'])

# Restore models
backbone.load_state_dict(checkpoint['backbone_state_dict'])
fc.load_state_dict(checkpoint['fc_state_dict'])

# Restore optimizer
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## Usage 


```shell
python bal_script.py
```

Example: With strategy selection, epoch, simulation, rounds, cold start and  learning rate

```shell
python bal_script.py --strategy bald --epochs 10 --sim 20 --rounds 15 --cold_start --lr 1e-6
```

Example: With data directory, backbone and fc path
```shell
python bal_script.py --data_dir /abc/123 --backbone_path /abd/123/backbone --fc_path /abc/123/fc
```

<hr>
