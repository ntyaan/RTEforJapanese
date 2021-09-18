import torch
from sklearn import metrics
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
from bert import read_train_data_replace
from const import Const


def make_encoding(MODEL_PATH, path, length):
    # トークナイザのロード
    tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_PATH)
    df = read_train_data_replace(path)
    dataset_for_loader = []
    for h, p in tqdm(zip(df.Hypothesis, df.Premise)):
        encoding = tokenizer(
            h,
            p,
            max_length=length,
            padding='max_length',
            truncation=True
        )
        encoding = {k: torch.tensor(v) for k, v in encoding.items()}
        dataset_for_loader.append(encoding)
    return list(df.Label), dataset_for_loader


def main():
    for i in range(10):
        print('----------', i, '----------')
        C = Const(i)
        model = BertForSequenceClassification.from_pretrained(
            C.MODEL_PATH
        )
        if C.USE_GPU:
            model.cuda()
        label_list, dataset_for_loader = make_encoding(
            C.MODEL_PATH, C.VALIDATION_PATH, C.ARG_DICT['max_train_length'])
        val_loader = DataLoader(
            dataset_for_loader,
            batch_size=C.ARG_DICT['eval_batch_size'],
            num_workers=C.ARG_DICT['num_works'])
        validation_outputs = []
        for batch in tqdm(val_loader):
            input_ids = batch['input_ids']
            token_type_ids = batch['token_type_ids']
            attention_mask = batch['attention_mask']
            if C.USE_GPU:
                batch['input_ids'] = input_ids.cuda()
                batch['token_type_ids'] = token_type_ids.cuda()
                batch['attention_mask'] = attention_mask.cuda()
            with torch.no_grad():
                output = model(**batch)
            scores = output.logits
            validation_outputs.extend(scores.max(axis=1).indices.tolist())

        accuracy = metrics.accuracy_score(
            label_list, validation_outputs)
        print(accuracy)
        cm = metrics.confusion_matrix(
            label_list, validation_outputs)
        print('正 負 負 負 正 負 負 負 正')
        print(cm.flatten())
        print(metrics.classification_report(
            label_list, validation_outputs))
        del model


if __name__ == '__main__':
    main()
