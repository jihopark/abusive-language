# Abusive Language

## Requirements
- python 3.4
- tensorflow 1.0
- keras 2.0 (for tf model layer construction)
- numpy
- sklearn (for splitting dataset)
- nltk (for ngram)
- pandas (for loading csv)
- re (for regex)
- jupyter notebook (for debugging) 
- tqdm (for displaying process)
- pickle (for saving python object)
- wordsegment, pyenchant (for preprocessing words)
- gensim (for word2vec)

## Scripts
1. `train.py`

## Modules
### data
- `preprocess.py`: clean the tweet
- `tokenizer.py`: tokenize text into characters/words
- `char.py`: helpers related to character features
- `word.py`: helpers related to word features
- `hybrid.py`: helpers related to char and word together
- `utils.py`: helpers related to dataset (splitting, batch generation, error analysis)

### model
- `char_cnn.py`: character-level convolutional neural network
- `word_cnn.py`: word-level convolutional neural network
- `hybrid.py`: word & char hybrid convolutional neural network

## See run results from Tensorboard
- after each run, the logs will be saved at `/logs/`. check command line log for the exact directory
- use `--log_dir` option to specify the log folder
- run `tensorboard --logdir=/logs/xxxxxx`
- if using remote server, ssh with portforwarding -L option. ex. `ssh -L 16006:127.0.0.1:6006 jhpark@remoteserver.hk` This option will forward remote 6006 port (default port for tensorboard) to localhost 16006.

## Notebooks
- `unshared_task_analysis/ipynb`: analysis of the Hate Speech Dataset
- `splitting_dataset.ipynb` : scripts for splitting dataset for experiment
- `baseline_lr.ipynb` : baseline ngram logistic regression experiments

## Datasets
1. `Hate Speech Dataset`: see `original/README.md`
2. `Pretrained word2vec`: please download it at `https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing` and put it in `data/`

## Python Style Guide
Please follow PEP 8.
https://www.python.org/dev/peps/pep-0008/
https://realpython.com/blog/python/vim-and-python-a-match-made-in-heaven/
