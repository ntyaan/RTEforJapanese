
import random
import torch
import numpy as np
import torch_optimizer as optim
import pandas as pd
from tqdm.auto import tqdm
import pytorch_lightning as pl
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
MAPPING = {'entailment': 0, 'contradiction': 1, 'neutral': 2}


class BertForSequenceClassification_pl(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.set_seed(hparams['seed'])

        # BERTのロード
        self.model = BertForSequenceClassification.from_pretrained(
            hparams['model_name'],
            num_labels=hparams['num_labels']
        )

        # トークナイザーの読み込み
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(
            hparams['model_name'], is_fast=True
        )

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # 学習データのミニバッチ(`batch`)が与えられた時に損失を出力する関数を書く。
    # batch_idxはミニバッチの番号であるが今回は使わない。
    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output.loss
        self.log('train_loss', loss)  # 損失を'train_loss'の名前でログをとる。
        return loss

    # 検証データのミニバッチが与えられた時に、
    # 検証データを評価する指標を計算する関数を書く。
    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss)  # 損失を'val_loss'の名前でログをとる。

    # 学習に用いるオプティマイザを返す関数を書く。
    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        print(self.parameters())
        return optim.RAdam(self.parameters(),
                           lr=self.hparams.hparams['learning_rate'])

    def setup(self, stage=None):
        """初期設定（データセットの読み込み）"""
        if stage == 'fit' or stage is None:
            self.train_dataset = self.get_three_dataset_for_loader(
                self.hparams.hparams['train_path'])

            self.val_dataset = self.get_three_dataset_for_loader(
                self.hparams.hparams['validaion_path'])

    def train_dataloader(self):
        """訓練データローダーを作成する"""
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.hparams['train_batch_size'],
                          drop_last=True, shuffle=True,
                          num_workers=self.hparams.hparams['num_works'])

    def val_dataloader(self):
        """バリデーションデータローダーを作成する"""
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.hparams['eval_batch_size'],
                          num_workers=self.hparams.hparams['num_works'])

    def get_three_dataset_for_loader(self, path):
        df = read_train_data_replace(path)
        # 各データの形式を整える
        dataset_for_loader = []
        for label, h, p in tqdm(zip(df.Label, df.Hypothesis, df.Premise)):
            encoding = self.tokenizer(
                h,
                p,
                max_length=self.hparams.hparams['max_train_length'],
                padding='max_length',
                truncation=True
            )
            encoding['labels'] = label
            encoding = {k: torch.tensor(v) for k, v in encoding.items()}
            dataset_for_loader.append(encoding)
        return dataset_for_loader


def read_train_data_replace(path):
    df = pd.read_csv(path, names=[
        'Label', 'Hypothesis', 'Premise'],  sep='\t')
    df.Label = df.Label.map(MAPPING)
    df.Hypothesis = df.Hypothesis.str.replace('。', '')
    df.Premise = df.Premise.str.replace('。', '')
    return df
