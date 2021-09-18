import random
import torch
import numpy as np
import pytorch_lightning as pl
import torch_optimizer as optim
from torch.utils.data import DataLoader
from totsv import TsvDataset

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.set_seed(self.hparams.seed)

        # 事前学習済みモデルの読み込み
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.hparams.model_name)

        # トークナイザーの読み込み
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.hparams.model_name, is_fast=True)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        """順伝搬"""
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

    def _step(self, batch):
        """ロス計算"""
        labels = batch['target_ids']

        # All labels set to -100 are ignored (masked),
        # the loss is only computed for labels in [0, ..., config.vocab_size]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch['source_ids'],
            attention_mask=batch['source_mask'],
            decoder_attention_mask=batch['target_mask'],
            labels=labels
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        """訓練ステップ処理"""
        loss = self._step(batch)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        """バリデーションステップ処理"""
        loss = self._step(batch)
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def configure_optimizers(self):
        """オプティマイザーとスケジューラーを作成する"""
        model = self.model
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                'weight_decay': self.hparams.weight_decay,
            },
            {
                'params': [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        optimizer = optim.RAdam(optimizer_grouped_parameters,
                                lr=self.hparams.learning_rate,
                                eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.t_total
        )

        return [optimizer], \
            [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]

    def setup(self, stage=None):
        """初期設定（データセットの読み込み）"""
        if stage == 'fit' or stage is None:
            train_dataset = TsvDataset(
                tokenizer=self.tokenizer,
                file_path=self.hparams.train_path,
                input_max_len=self.hparams.max_train_length,
                target_max_len=self.hparams.max_target_length)
            self.train_dataset = train_dataset

            val_dataset = TsvDataset(
                tokenizer=self.tokenizer,
                file_path=self.hparams.validaion_path,
                input_max_len=self.hparams.max_train_length,
                target_max_len=self.hparams.max_target_length)
            self.val_dataset = val_dataset

            self.t_total = (
                (len(train_dataset) //
                 (self.hparams.train_batch_size
                 * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
            )

    def train_dataloader(self):
        """訓練データローダーを作成する"""
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.train_batch_size,
                          drop_last=True, shuffle=True,
                          num_workers=self.hparams.num_works)

    def val_dataloader(self):
        """バリデーションデータローダーを作成する"""
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.eval_batch_size,
                          num_workers=self.hparams.num_works)
