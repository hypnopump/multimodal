# MultiModal


### Installation

```bash
# install packages and download
pip install -r requirements.txt
python3 download_data.py  
# OR [if want faster, at your own risk] python3 download_data_parallel.py 

# now process the files
python3 binaries_big_file.py

# now train the model
python3 train_autoregressive.py --config configs/local_config.yaml
```