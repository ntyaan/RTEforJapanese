import pandas as pd

TRAIN_NAME = './original/train_w_filtering.tsv'
MAPPING_LIST = ['entailment', 'contradiction', 'neutral']
MAPPING = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
RANDOM_STATE = 42
SPLIT_NUM = 8800


class SplitDataset:

    def __init__(self) -> None:
        self.read_train_data_replace()
        self.get_label()
        self.get_label_len()

    def read_train_data_replace(self):
        self.df = pd.read_csv(TRAIN_NAME, names=[
            'Label', 'Hypothesis', 'Premise'],  sep='\t')

    def get_label(self):
        self.df_entailment = (
            self.df[self.df.Label == 'entailment']
            .sample(frac=1.0, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
        self.df_contradiction = (
            self.df[self.df.Label == 'contradiction']
            .sample(frac=1.0, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
        self.df_neutral = (
            self.df[self.df.Label == 'neutral']
            .sample(frac=1.0, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )

    def get_label_len(self):
        self.df_len = len(self.df)
        self.df_entailment_len = len(self.df_entailment)
        self.df_contradiction_len = len(self.df_contradiction)
        self.df_neutral_len = len(self.df_neutral)
        print(self.df_len, self.df_entailment_len,
              self.df_contradiction_len, self.df_neutral_len)

    def make_df(self, start=0, end=SPLIT_NUM):
        df = pd.concat(
            [self.df_entailment[start:end],
             self.df_contradiction[start:end]]
        ).reset_index(drop=True)
        df = pd.concat(
            [df, self.df_neutral[start:end]]
        ).reset_index(drop=True)
        return df

    def make_df_end(self, start_e, start_c, start_n):
        print(start_e, start_c, start_n)
        df = pd.concat(
            [self.df_entailment[start_e:],
             self.df_contradiction[start_c:]]
        ).reset_index(drop=True)
        df = pd.concat(
            [df, self.df_neutral[start_n:]]
        ).reset_index(drop=True)
        return df

    def to_tsv(self, f, df):
        df.to_csv(f, header=False,
                  index=False, sep='\t', encoding='utf-8')

    def main(self):
        start = 0
        end = SPLIT_NUM
        count = int(self.df_len / (SPLIT_NUM * 3 * 2))
        for c in range(count):
            for i in range(2):
                if i == 0:
                    validation_start = start
                    validation_end = end
                    df = self.make_df(validation_start, validation_end)
                    self.to_tsv('./validation/validation' +
                                str(c) + '.tsv', df)
                # elif i == 1:
                #     test_start = validation_end
                #     test_end = test_start + SPLIT_NUM
                #     df = self.make_df(test_start, test_end)
                #     self.to_tsv('./test/test' +
                #                 str(c) + '.tsv', df)
                elif i == 1:
                    print(start, end)
                    train_start = validation_end
                    train_end = train_start + SPLIT_NUM
                    if c == count - 1:
                        df = self.make_df_end(
                            train_start,
                            train_start,
                            train_start,
                        )
                    else:
                        df = self.make_df(train_start, train_end)
                    self.to_tsv('./train/train' + str(c) + '.tsv', df)
                    start = train_end
                    end = start + SPLIT_NUM


if __name__ == '__main__':
    SplitDataset().main()
