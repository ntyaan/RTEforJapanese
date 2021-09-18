import re
from torch.utils.data import Dataset
MAPPING = {'entailment': 0, 'contradiction': 1, 'neutral': 2}


class TsvDataset(Dataset):
    def __init__(self, tokenizer, file_path,
                 input_max_len=128, target_max_len=128):
        self.file_path = file_path

        self.input_max_len = input_max_len
        self.target_max_len = target_max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]['input_ids'].squeeze()
        target_ids = self.targets[index]['input_ids'].squeeze()

        source_mask = self.inputs[index]['attention_mask'].squeeze()
        target_mask = self.targets[index]['attention_mask'].squeeze()

        return {'source_ids': source_ids, 'source_mask': source_mask,
                'target_ids': target_ids, 'target_mask': target_mask}

    def _make_record(self, title, body, genre_id):
        # ニュース分類タスク用の入出力形式に変換する。
        input = f'{title} {body}'
        target = f'{genre_id}'
        return input, target

    def _build(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                assert len(line) == 3
                assert len(line[0]) > 0
                assert len(line[1]) > 0
                assert len(line[2]) > 0

                tokenized_inputs = self.tokenizer(
                    re.sub(r'[\u3000 \t]', '', line[1]).replace('。', ''),
                    re.sub(r'[\u3000 \t]', '', line[2]).replace('。', ''),
                    max_length=self.input_max_len, truncation=True,
                    padding='max_length', return_tensors='pt'
                )

                tokenized_targets = self.tokenizer(
                    str(MAPPING[line[0]]),
                    max_length=self.target_max_len, truncation=True,
                    padding='max_length', return_tensors='pt'
                )

                self.inputs.append(tokenized_inputs)
                self.targets.append(tokenized_targets)


class TsvDatasetForMainTest(Dataset):
    def __init__(self, tokenizer, file_path,
                 input_max_len=128):
        self.file_path = file_path

        self.input_max_len = input_max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]['input_ids'].squeeze()
        source_mask = self.inputs[index]['attention_mask'].squeeze()

        return {'source_ids': source_ids, 'source_mask': source_mask}

    def _make_record(self, title, body, genre_id):
        # ニュース分類タスク用の入出力形式に変換する。
        input = f'{title} {body}'
        target = f'{genre_id}'
        return input, target

    def _build(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                assert len(line) == 2
                assert len(line[0]) > 0
                assert len(line[1]) > 0

                tokenized_inputs = self.tokenizer(
                    re.sub(r'[\u3000 \t]', '', line[0]).replace('。', ''),
                    re.sub(r'[\u3000 \t]', '', line[1]).replace('。', ''),
                    max_length=self.input_max_len, truncation=True,
                    padding='max_length', return_tensors='pt'
                )

                self.inputs.append(tokenized_inputs)
