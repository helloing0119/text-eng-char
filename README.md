## Dependencies

The code requires the following dependencies to be installed:
- tensorFlow
- pandas
- python >= 3.9.0
- pip

You can install the dependencies by running the following command:

```
pip install tensorflow pandas
```

## Usage
```
python training.py
```
or you can run with aruguments like...
```
python training.py --epochs 100 --use-random-seed False --seed 777 --learning-rate 0.001 --save-path "./models/result" --hw-accelerator "cpu" --csv-path "./text_dataset/english.csv" --img-dir "./text_dataset/" --csv-img-col "image" --csv-label-col "label" --num-classes 62
```

The script uses command-line arguments to customize its behavior. Here is a description of the available arguments:

- `--epochs`: Number of epochs to train the model (default: 100).
- `--use-random-seed`: Whether to use a random seed for generating the dataset (default: False).
- `--seed`: Seed for generating the dataset, a positive value (default: 777).
- `--learning-rate`: Learning rate for the model training (default: 0.001). Must be between 0.00001 and 0.1.
- `--save-path`: File path to save the trained model (default: "./models/result").
- `--hw-accelerator`: Hardware accelerator to use, either "cpu" or "gpu" (default: "cpu").
- `--csv-path`: File path to the label CSV file (default: "./text_dataset/english.csv").
- `--img-dir`: Directory path for the images referenced in the CSV (default: "./text_dataset/").
- `--csv-img-col`: Column name for the image filenames in the CSV (default: "image").
- `--csv-label-col`: Column name for the label values in the CSV (default: "label").
- `--num-classes`: Number of classes for the labels (default: 62).
