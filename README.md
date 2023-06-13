## Dependencies

The code requires the following dependencies to be installed:
- tensorFlow==2.9.0
- pandas
- python >= 3.9.0
- pip
- cv2

You can install the dependencies by running the following command:

```
pip install tensorflow==2.9.0 pandas
```
## Dir & Files
`training.py` : python script for learning
`text_dataset/` : dataset for learning
`test_dataset/` : dataset for test
`test_dataset/parse_data.py` : generate label data
`test_dataset/modify_image.py` : modify test dataset, adding noise, tilting image, transfering center

## Recomended setting
--epoch 5 --use-random-seed --test-len 100

## Usage
```
python training.py
```
or you can run with aruguments like...
```
python training.py --epochs 100 --batch-size 32 --use-random-seed False --seed 777 --learning-rate 0.001 --save-path "./models/result" --hw-accelerator "cpu" --csv-path "./text_dataset/english.csv" --img-dir "./text_dataset/" --test-csv-path "./test_dataset/test_data.csv" --test-img-dir "./test_dataset/" --test-len 100 --csv-img-col "image" --csv-label-col "label" --num-classes 62
```

recommended
```
python training.py --epoch 5 --use-random-seed --test-len 100
```

The script uses command-line arguments to customize its behavior. Here is a description of the available arguments:

- `--epochs`: Number of epochs to train the model (default: 100).
- `--batch-size`: Batch size to train the model (default: 32).
- `--use-random-seed`: Whether to use a random seed for generating the dataset (default: False).
- `--seed`: Seed for generating the dataset, a positive value (default: 777).
- `--learning-rate`: Learning rate for the model training (default: 0.001). Must be between 0.0001 and 0.1.
- `--save-path`: File path to save the trained model (default: "./models/result").
- `--hw-accelerator`: Hardware accelerator to use, either "cpu" or "gpu" (default: "cpu").

- `--csv-path`: File path to the label CSV file (default: "./text_dataset/english.csv").
- `--img-dir`: Directory path for the images referenced in the CSV (default: "./text_dataset/").
- `--test-csv-path`: File path to the test label CSV file (default: "./test_dataset/test_data.csv").
- `--test-img-dir`: Directory path for the images referenced in the test CSV (default: "./test_dataset/").
- `--test-len`: Number of samples to evaluate the model (default: 100).
- `--csv-img-col`: Column name for the image filenames in the CSV (default: "image").
- `--csv-label-col`: Column name for the label values in the CSV (default: "label").
- `--num-classes`: Number of classes for the labels (default: 62).
