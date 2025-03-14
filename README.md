# ACS-project-Wormhole
GitHub repo for ACS project "Realisation of specialized ML/HPC algorithms on Tenstorrent Wormhole"

### Prerequisites
g++, python(with sklearn, pandas, json packages installed), c++(install nlohmann)

### Data description
To begin with, we provided some example csv files based on iris dataset, which our decision tree and random forest algorithms specify. 

In train.csv there is a dataset which is used for training a tree/forest model in Python. 

### Compilation
Firstly, to train a tree run this:
```{bash}
python3 tree.py
```
and to train forest:
```{bash}
python3 forest.py
```

It builds json files for tree and forest weights respectively, which is used in our cpp file for classificating given samples. To compile cpp file, run this in terminal:

```{bash}
g++ -o test decision_tree.cpp -O3
```
or 
```{bash}
g++ -o test decision_forest.cpp -O3
```

### Running
To run a programe, pastw thia in terminal:
```{bash}
./test
```
Tree predicts classes for the given instances. Forest predicts class based on maximum vote.
