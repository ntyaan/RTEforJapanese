import torch
import pytorch_lightning as pl
TEST_VERSION = 'v21'


class Const:
    # コンストラクタの定義
    def __init__(self, arg):
        # 事前学習済みモデル
        self.PRETRAINED_MODEL_NAME = 'cl-tohoku/bert-base-japanese-v2'

        # チェックポイントパス
        self.CHECKPOINT_PATH = './content/checkpoints/' + \
            TEST_VERSION+'/' + str(arg)
        # モデル保存パス
        self.MODEL_PATH = './content/model/'+TEST_VERSION+'/' + str(arg)
        # データセットパス
        self.TRAIN_PATH = '../dataset/train/train' + \
            str(arg) + '.tsv'
        self.VALIDATION_PATH = '../dataset/validation/validation' + \
            str(arg) + '.tsv'
        # GPU利用有無
        self.USE_GPU = torch.cuda.is_available()

        # 各種ハイパーパラメータ
        self.ARG_DICT = dict(
            train_path=self.TRAIN_PATH,
            validaion_path=self.VALIDATION_PATH,
            model_name=self.PRETRAINED_MODEL_NAME,
            learning_rate=1e-5,
            weight_decay=0.0,
            adam_epsilon=1e-8,
            warmup_steps=0,
            gradient_accumulation_steps=1,
            n_gpu=1 if self.USE_GPU else 0,
            early_stop_callback=False,
            fp_16=False,
            opt_level='O1',
            max_grad_norm=1.0,
            seed=42,
            max_train_length=142,
            max_target_length=3,
            train_batch_size=32,
            eval_batch_size=256,
            num_train_epochs=3,
            num_works=0,
            num_labels=3
        )

        self.CP_CB = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_weights_only=True,
            dirpath=self.CHECKPOINT_PATH,
        )

        # Trainer パラメータ
        self.TRAIN_PARAMS = dict(
            accumulate_grad_batches=self.ARG_DICT
            ['gradient_accumulation_steps'],
            gpus=self.ARG_DICT['n_gpu'],
            max_epochs=self.ARG_DICT['num_train_epochs'],
            precision=16 if self.ARG_DICT['fp_16'] else 32,
            amp_level=self.ARG_DICT['opt_level'],
            gradient_clip_val=1.0,
            callbacks=[self.CP_CB]
        )
