import pytorch_lightning as pl
from const import Const
from bert import BertForSequenceClassification_pl

if __name__ == '__main__':
    for i in range(10):
        print('----------', i, '----------')
        C = Const(i)

        # 学習の方法を指定
        trainer = pl.Trainer(**C.TRAIN_PARAMS)

        # PyTorch Lightningモデルのロード
        model = BertForSequenceClassification_pl(C.ARG_DICT)

        # ファインチューニングを行う。
        trainer.fit(model)

        best_model_path = C.CP_CB.best_model_path  # ベストモデルのファイル
        print('ベストモデルのファイル: ', best_model_path)

        # PyTorch Lightningモデルのロード
        model = BertForSequenceClassification_pl.load_from_checkpoint(
            best_model_path
        )

        # Transformers対応のモデルを保存
        model.model.save_pretrained(C.MODEL_PATH)
        model.tokenizer.save_pretrained(C.MODEL_PATH)
