import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader


class JobsDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_len=512):
        self.len = len(X)
        self.data = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len
        self._label_encode()

    def _label_encode(self):
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)

    def __getitem__(self, index):
        title = str(self.data.title[index])
        title = " ".join(title.split())
        description = str(self.data.description[index])
        description = " ".join(description.split())
        inputs = self.tokenizer.encode_plus(
            text=title,
            text_pair=description,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.y[index], dtype=torch.long)
        }

    def __len__(self):
        return self.len


def get_data_loader(X_train, X_valid, y_train, y_valid, tokenizer, batch_size=16, num_workers=1):
    training_set = JobsDataset(X_train, y_train, tokenizer, max_len=512)
    validation_set = JobsDataset(X_valid, y_valid, tokenizer, max_len=512)
    train_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'num_workers': num_workers
                    }

    test_params = {'batch_size': batch_size,
                   'shuffle': True,
                   'num_workers': num_workers
                   }

    training_loader = DataLoader(training_set, **train_params)
    validation_loader = DataLoader(validation_set, **test_params)
    return training_loader, validation_loader
