class Vocabulary:
    def __init__(self, token_to_idx=None):
        if token_to_idx is None:
            token_to_idx = {}
        self.token_to_idx = token_to_idx

        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

    def add_token(self, token):
        if token in self.token_to_idx:
            index = self.token_to_idx[token]
        else:
            index = len(self.token_to_idx)
            self.token_to_idx[token] = index
            self.idx_to_token[index] = token
        return index

    def add_tokens(self, tokens):
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        return self.token_to_idx[token]

    def lookup_index(self, index):
        if index not in self.idx_to_token:
            raise KeyError(f"the index ({index}) is not in the Vocabulary")
        return self.idx_to_token[index]

    def __str__(self):
        return f"<Vocabulary(size={len(self)})>"

    def __len__(self):
        return len(self.token_to_idx)

    def to_serializable(self):
        return {"token_to_idx": self.token_to_idx}

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)


class SequenceVocabulary(Vocabulary):
    "Wraps around Vocab to add start and end tokens"

    def __init__(
        self,
        token_to_idx: dict = None,
        unk_token: str = "<UNK>",
        begin_seq_token: str = "<BEGIN>",
        end_seq_token: str = "<END>",
    ):
        super().__init__(token_to_idx)

        self.unk_token = unk_token
        self.begin_seq_token = begin_seq_token
        self.end_seq_token = end_seq_token

        self.unk_index = self.add_token(self.unk_token)
        self.begin_seq_index = self.add_token(self.begin_seq_token)
        self.end_seq_index = self.add_token(self.end_seq_token)

        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

    def lookup_token(self, token):
        return self.token_to_idx.get(token, self.unk_index)

    def lookup_index(self, index):
        if index not in self.idx_to_token:
            raise KeyError("the index ({index}) is not in the SequenceVocabulary")
        return self.idx_to_token[index]

    def __str__(self):
        return "<SequenceVocabulary(size={len(self.token_to_idx)})>"

    def __len__(self):
        return len(self.token_to_idx)

    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update(
            {
                "unk_token": self.unk_token,
                "begin_seq_token": self.begin_seq_token,
                "end_seq_token": self.end_seq_token,
            }
        )
        return contents
