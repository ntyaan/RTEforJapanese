import pandas as pd
from const import Const
from sklearn import metrics
from transformers import T5ForConditionalGeneration, T5Tokenizer
from totsv import TsvDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

RESULT = './content/result/00/'
MAPPING_LIST = ['entailment', 'contradiction', 'neutral']
RESULT = './content/result/00/'


def write_result(file_path, label_list, path):
    df = pd.read_csv(file_path, names=[
        'Labels', 'Hypothesis', 'Premise'],  sep='\t')
    for i, j in enumerate(label_list):
        df.iat[i, 0] = MAPPING_LIST[int(j)]
    df = df.set_index('Labels')
    df.to_csv(path, sep='\t', header=None)


def get_outputs(C):
    tokenizer = T5Tokenizer.from_pretrained(C.MODEL_PATH, is_fast=True)
    trained_model = T5ForConditionalGeneration.from_pretrained(
        C.MODEL_PATH)
    if C.USE_GPU:
        trained_model.cuda()
    val_dataset = TsvDataset(
        tokenizer=tokenizer,
        file_path=C.VALIDATION_PATH,
        input_max_len=C.ARG_DICT['max_train_length'],
        target_max_len=C.ARG_DICT['max_target_length'])
    val_loader = DataLoader(
        val_dataset, batch_size=C.ARG_DICT['eval_batch_size'],
        num_workers=C.ARG_DICT['num_works'])
    trained_model.eval()
    validation_outputs = []
    validation_targets = []
    for batch in tqdm(val_loader):
        input_ids = batch['source_ids']
        input_mask = batch['source_mask']
        if C.USE_GPU:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
        outs = trained_model.generate(
            input_ids=input_ids,
            attention_mask=input_mask,
            max_length=C.ARG_DICT['max_target_length'],
            return_dict_in_generate=True,
            output_scores=True)
        dec = [tokenizer.decode(ids, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
               for ids in outs.sequences]
        target = [tokenizer.decode(ids, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
                  for ids in batch["target_ids"]]
        validation_outputs.extend(dec)
        validation_targets.extend(target)

    accuracy = metrics.accuracy_score(
        validation_targets, validation_outputs)
    print(accuracy)
    cm = metrics.confusion_matrix(
        validation_targets, validation_outputs)
    print('正 負 負 負 正 負 負 負 正')
    print(cm.flatten())
    print(metrics.classification_report(
        validation_targets, validation_outputs))

    return validation_outputs


if __name__ == '__main__':
    for i in range(10):
        print('----------', i, '----------')
        C = Const(i)
        label_list = get_outputs(C)
        write_result(C.VALIDATION_PATH, label_list, RESULT+str(i)+'.tsv')
