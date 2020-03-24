# CoveoDataChallenge
## Setup 
* Install Virtual Env
```
pip install virtualenv
```
* Create a Virtual Env
```
virtualenv $name$
source venv/bin/activate
```
* Install requirements:
```
pip install -r requirements.txt
```
* Set up Python Path to include src

## Sample argparse Statements
```
python main.py $data_path$ 1
python main.py $data_path$ 2 --path_to_previous $path_to_previous$ --num_predictions 3
python main.py $data_path$ 3 --path_to_previous $path_to_previous$ --num_predictions 3 --country UK
```

