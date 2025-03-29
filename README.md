# ACS-project-Wormhole
GitHub repo for ACS project "Realisation of specialized ML/HPC algorithms on Tenstorrent Wormhole"

### Prerequisites
- g++
- Python 3.9.6+
- C++ (install nlohmann)

Python dependencies:
```bash
pip3 install -r requirements.txt
```

C++ dependencies Ubuntu:
```bash
sudo apt install libbenchmark-dev nlohmann-json-dev
```

C++ dependencies MacOS:
```bash
brew install google-benchmark nlohmann-json
```

### Data description
To begin with, we provided some example csv files based on iris dataset, which our decision tree and random forest algorithms specify. 

In train.csv there is a dataset which is used for training a tree/forest model in Python. 

### Compilation
Firstly, to train a tree run this:
```{bash}
python3 utils/tree.py
```
and to train forest:
```{bash}
python3 utils/forest.py
```

Compile project:
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ../
make -j8
```

### Running
To run a programe:
```{bash}
./test_tree
./test_forest
./benchmark_wormhole
```
Tree predicts classes for the given instances. Forest predicts class based on maximum vote.
