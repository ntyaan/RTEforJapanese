import pytorch_lightning as pl
from const import Const
from t5 import T5FineTuner

if __name__ == '__main__':
    for i in range(10):
        print('----------', i, '----------')
        C = Const(i)
        # 転移学習の実行
        model = T5FineTuner(C.ARG_DICT)
        trainer = pl.Trainer(**C.TRAIN_PARAMS)

        trainer.fit(model)

        best_model_path = C.CP_CB.best_model_path  # ベストモデルのファイル
        print('ベストモデルのファイル: ', best_model_path)

        # PyTorch Lightningモデルのロード
        model = T5FineTuner.load_from_checkpoint(
            best_model_path
        )

        # Transformers対応のモデルを保存
        model.model.save_pretrained(C.MODEL_PATH)
        model.tokenizer.save_pretrained(C.MODEL_PATH)

        del model
