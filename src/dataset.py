import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from torch.utils.data import Dataset, DataLoader
from vocab import SequenceVocabulary
from vectorizer import *


class EHRDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vectorizer: EHRCountVectorizer,
        include_features=True,
        include_timedels=True,
        num_workers=8,
    ):
        self.df = df
        self.vectorizer = vectorizer
        self.include_features = include_features
        self.include_timedels = include_timedels
        self.num_workers = num_workers

        # Data splits
        self.train_df = self.df[self.df.split == "train"]
        self.train_size = len(self.train_df)
        self.val_df = self.df[self.df.split == "val"]
        self.val_size = len(self.val_df)
        self.test_df = self.df[self.df.split == "test"]
        self.test_size = len(self.test_df)
        self.ho_df = self.df[self.df.split == "heldout"]
        self.ho_size = len(self.ho_df)
        self.lookup_dict = {
            "train": (self.train_df, self.train_size),
            "val": (self.val_df, self.val_size),
            "test": (self.test_df, self.test_size),
            "heldout": (self.ho_df, self.ho_size),
        }
        self.set_split("train")

    def set_split(self, split="train"):
        self.target_split = split
        self.target_df, self.target_size = self.lookup_dict[split]

    @classmethod
    def load_dataset_and_make_vectorizer(cls, df):
        train_df = df[df.split == "train"]
        return cls(df, EHRCountVectorizer.from_dataframe(train_df))

    def generate_batches(
        self, batch_size: int, collate_fn, shuffle=True, drop_last=False, device="cpu"
    ):
        dataloader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
        )

        for data_dict in dataloader:
            out_data_dict = {}
            for name, _ in data_dict.items():
                out_data_dict[name] = data_dict[name]
            yield out_data_dict

    def get_num_batches(self, batch_size):
        return len(self) // batch_size

    def __getitem__(self, index):
        # row is each row in the dataframe
        row = self.target_df.iloc[index]
        diag_for_patient, patient_num_visits = self.vectorizer.vectorize(row.X_seq)
        target_y = row.target_y
        gender, age, timedels = None, None, []

        if self.include_timedels:
            timedels = np.array(row.days_from_prev.split(";")).astype(np.float)

        if self.include_features:
            gender = int(row.sukup)
            age = row.ika
            prev_num_visits = row.prev_num_visits
            time_delta_mean = row.time_delta_mean

        krypht = row.krypht
        return {
            "x_seq": diag_for_patient,
            "seq_length": patient_num_visits,
            "age": age,
            "gender": gender,
            "prev_num_visits": prev_num_visits,
            "time_delta_mean": time_delta_mean,
            "timedels": timedels,
            "krypht": krypht,
            "target_y": target_y,
        }

    def get_vectorizer(self):
        return self.vectorizer

    def __str__(self):
        return "<Dataset(split={0}, size={1})".format(
            self.target_split, self.target_size
        )

    def __len__(self):
        return self.target_size
