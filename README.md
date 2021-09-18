# RTE for Japanese
Recognizing Textual Entailment for Japanese with BERT and t5

## Usage
* Download original dataset from https://nlp.ist.i.kyoto-u.ac.jp/index.php?%E6%97%A5%E6%9C%AC%E8%AA%9ESNLI%28JSNLI%29%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88
* Put `train_w_filtering.tsv` to `./dataset/original` and run `split_dataset.py`
* Set `const.py` and fine tuning by `fit.py`
