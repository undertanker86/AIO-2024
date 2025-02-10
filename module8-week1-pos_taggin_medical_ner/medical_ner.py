import os
from typing import List, Dict, Tuple

class Preprocessing_Macrobat:
    def __init__(self, dataset_folder, tokenizer):
        self.file_ids = [f.split(".")[0] for f in os.listdir(dataset_folder) if f.endswith('.txt')]
        self.text_files = [f + ".txt" for f in self.file_ids]
        self.anno_files = [f + ".ann" for f in self.file_ids]
        self.num_samples = len(self.file_ids)
        self.texts: List[str] = []

        for i in range(self.num_samples):
            file_path = os.path.join(dataset_folder, self.text_files[i])
            with open(file_path, 'r') as f:
                self.texts.append(f.read())

        self.tags: List[Dict[str, str]] = []
        for i in range(self.num_samples):
            file_path = os.path.join(dataset_folder, self.anno_files[i])
            with open(file_path, 'r') as f:
                text_bound_ann = [f.split("\n") for t in f.read().split("\n") if t.startswith("T")]
            text_bound_lst = []
            for text_b in text_bound_ann:
                label = text_b[1].split("\t")
                try:
                    tag = {
                        "text": text_b[-1],
                        "label": label[0],
                        "start": label[1],
                        "end": label[2]
                    }
                except:
                    pass
            self.tags.append(text_bound_lst)
        self.tokenizer = tokenizer

    def process(self) -> Tuple[List[List[str]], List[List[str]]]:
        input_texts = []
        input_labels = []

        for idx in range(self.num_samples):
            full_text = self.texts[idx]
            tags = self.tags[idx]
            label_offset = []
            continuous_label_offset = []
            for tag in tags:
                offset = list(range(int(tag["start"]), int(tag["end"])))
                label_offset.append(offset)
                continuous_label_offset.extend(offset)
            all_offset = list(range(len(full_text)))
            zero_offset = [offset for offset in all_offset if offset not in continuous_label_offset]
            zero_offset = Preprocessing_Macrobat.find_continuous_ranges(zero_offset)

            self.tokens = []
            self.labels = []
            self._merge_offset(full_text, tags, zero_offset, label_offset)

            assert len(self.tokens) == len(self.labels), "Length of tokens and labels are not equal"

            input_texts.append(self.tokens)
            input_labels.append(self.labels)

        return input_texts, input_labels

    def _merge_offset(self, full_text, tags, zero_offset, label_offset):
        j = 0
        while i < len(zero_offset) and j < len(label_offset):
            if zero_offset[i][0] < label_offset[j][0]:
                self._add_zero(full_text, zero_offset, i)
                i += 1
            else:
                self._add_label(full_text, label_offset, j, tags)
                j += 1
        while i < len(zero_offset):
            self._add_zero(full_text, zero_offset, i)
            i += 1
        while j < len(label_offset):
            self._add_label(full_text, label_offset, j, tags)
            j += 1

    def _add_zero(self, full_text, offset, index):
        start, end = offset[index] if len(offset[index]) > 1 else (offset[index][0], offset[index][0]+1)
        text = full_text[start:end]
        text_tokens = self.tokenizer.tokenize(text)
        self.tokens.extend(text_tokens)
        self.labels.extend(["O"]*len(text_tokens))

    def _add_label(self, full_text, offset, index, tags):
        start, end = offset[index] if len(offset[index]) > 1 else (offset[index][0], offset[index][0]+1)
        text = full_text[start:end]
        text_tokens = self.tokenizer.tokenize(text)
        self.tokens.extend(text_tokens)
        self.labels.extend([f"B-{tags[index]['label']}"] + [f"I-{tags[index]['label']}"]*(len(text_tokens)-1))

    @staticmethod
    def build_label2id(tokens: List[str]) -> Dict[str, int]:
        label2id = {}
        id_counter = 0
        for token in tokens:
            if token not in label2id:
                label2id[token] = id_counter
                id_counter += 1
        return label2id

    @staticmethod
    def find_continuous_ranges(data: List[int]):
        if not data:
            return []
        ranges = []
        start = number = data[0]
        for number in data[1:]:
            if number != prev + 1:
                ranges.append(list(range(start, prev + 1)))
                start = number
            prev = number
        ranges.append(list(range(start, prev + 1)))
        return ranges

# Preprocessing
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
dataset_folder = "*./MACCROBATA2018*"
Macrobot_builder = Preprocessing_Macrobat(dataset_folder, tokenizer)
input_texts, input_labels = Macrobot_builder.process()

label2id = Preprocessing_Macrobat.build_label2id(input_labels)
id2Label = {v: k for k, v in label2id.items()}

# Split
from sklearn.model_selection import train_test_split

inputs_train, inputs_val, labels_train, labels_val = train_test_split(
    input_texts,
    input_labels,
    test_size=0.2,
    random_state=42
)


import torch
from torch.utils.data import Dataset

MAX_LEN = 512

class NER_Dataset(Dataset):
    def __init__(self, input_texts, input_labels, tokenizer, label2id, max_len=MAX_LEN):
        super().__init__()
        self.tokens = input_texts
        self.labels = input_labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        input_token = self.tokens[idx]
        label_token = [self.label2id[label] for label in self.labels[idx]]
        input_token = self.tokenizer.convert_tokens_to_ids(input_token)
        attention_mask = [1] * len(input_token)

        input_ids = self.pad_and_truncate(input_token, pad_id=self.tokenizer.pad_token_id)
        labels = self.pad_and_truncate(label_token, pad_id=0)
        attention_mask = self.pad_and_truncate(attention_mask, pad_id=0)

        return {
            "input_ids": torch.as_tensor(input_ids),
            "labels": torch.as_tensor(labels),
            "attention_mask": torch.as_tensor(attention_mask)
        }

    def pad_and_truncate(self, inputs: List[int], pad_id: int):
        if len(inputs) < self.max_len:
            padded_inputs = inputs + [pad_id] * (self.max_len - len(inputs))
        else:
            padded_inputs = inputs[:self.max_len]
        return padded_inputs

    def label2id(self, labels: List[str]):
        return [self.label2id[label] for label in labels]

train_set = NER_Dataset(inputs_train, labels_train, tokenizer, label2id)
val_set = NER_Dataset(inputs_val, labels_val, tokenizer, label2id)


import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mask = labels != 0
    predictions = np.argmax(predictions, axis=-1)
    return accuracy.compute(predictions=predictions[mask], references=labels[mask])


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="out_dir",
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    optim="adamw_torch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()


test_sentence = """48 year-old female presented with vaginal bleeding and abnormal Pap smears. Upon diagnosis of invasive non-keratinizing SCC of the cervix, she underwent a radical hysterectomy with salpingo-oophorectomy which demonstrated positive spread to the pelvic lymph nodes and the parametrium. Pathological examination revealed that the tumour also extensively involved the lower uterine segment."""
input = torch.as_tensor([tokenizer.convert_tokens_to_ids(test_sentence.split())])
input = input.to("cuda")

# prediction
outputs = model(input)
_, preds = torch.max(outputs.logits, -1)
preds = preds[0].cpu().numpy()

# decode
for token, pred in zip(test_sentence.split(), preds):
    print(f"{token}\t{id2label[pred]}")
