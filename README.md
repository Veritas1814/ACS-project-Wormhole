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

### Build dependencies
```bash
git submodule update --init --recursive
git submodule foreach --recursive 'git lfs fetch --all && git lfs pull'
```

### Compilation
Firstly, to train a tree run this:
```{bash}
python tree.py <train_file_csv> <test_file_csv> <tree_output_file_json> <predictions_output_file_tree_csv>
```
and to train forest:
```{bash}
python forest.py <train_file_csv> <test_file_csv> <forest_output_file_json> <predictions_output_file_forest_csv>
```

Compile project:
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_WORMHOLE_DECISION_TREE=ON ../  # You can compile without WH if needed
make -j$(nproc)
```

### Export those environment variables before running WH code: 
```bash
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(realpath ./third-party/tt-metal/)
```

### Running
To run a programe:
```{bash}
./test_tree
./test_forest
./benchmark_wormhole <tree_output_file_json> <test_file_csv>
sudo ./plot_result <tree_output_file_json> <test_file_csv> --benchmark_format=json --benchmark_out=benchmark_results.json
cd ..
python utils/plot_benchmarks.py    
```
The script will generate the following files in the data folder:

data/benchmark_results.csv <br>
data/duration_plot.png – A plot of the average execution time per prediction. <br>
data/iterations_plot.png – A plot of the total number of iterations.

Tree predicts classes for the given instances. Forest predicts class based on maximum vote.
