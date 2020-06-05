import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from torch.utils.data import Dataset, DataLoader
from vocab import SequenceVocabulary


class EHRCountVectorizer:
    """
    Vectorizer to convert a sequence of ICD10 code to vectorized arrays
    In this specific instance, output is not a sequence. Look at MIMICVectorizer for
    a more general implementation with sequence as output as well.
    """

    def __init__(self, diagnoses_vocab: SequenceVocabulary):
        self.diagnoses_vocab = diagnoses_vocab
        self.diagnoses_vocab_len = len(self.diagnoses_vocab)

    def vectorize(self, diagnoses: str):
        """
        Convert the diagnoses sequence to a list of indeces based on the
        vocabulary object. 
        This method also handles the padding of the "inner-list", i.e.,
        the number of diagnoses per visit. Padding the number of visits sequence
        is handled by the collate function in trainer.
        
        Args:
          diagnoses: String of diagnoses for a patient, each visit 
                     separated by ';', diagnoses per 
                     visit separated by space

        Returns:
          diag_for_patient_padded: ndarray of shape 
                                   [num_visits, max_num_diagnoses_per_visit] 
          patient_num_visits: number of visits for this patient.
        """
        visits = [visit for visit in diagnoses.split(";")]
        patient_number_of_visits = len(visits)
        item_per_visit = []
        max_items_per_visit_length = 0
        for visit in visits:
            items_per_visit_i = [
                self.diagnoses_vocab.lookup_token(token) for token in visit.split(" ")
            ]
            items_per_visit_i_length = len(items_per_visit_i)
            if max_items_per_visit_length < items_per_visit_i_length:
                max_items_per_visit_length = items_per_visit_i_length
            item_per_visit.append(items_per_visit_i)

        item_per_visit_padded = np.zeros(
            (patient_number_of_visits, max_items_per_visit_length), dtype=np.int64
        )

        for i, visit in enumerate(item_per_visit):
            item_per_visit_padded[i, : len(visit)] = visit

        return item_per_visit, patient_number_of_visits

    @classmethod
    def from_dataframe(cls, df, colname="X_seq"):
        diagnoses_vocab = SequenceVocabulary()
        for _, row in df.iterrows():
            diagnoses_vocab.add_tokens(row[f"{colname}"].replace(";", " ").split(" "))

        print(f"Corpus has {len(diagnoses_vocab)} unique tokens")
        return cls(diagnoses_vocab)
