# import library
from typing import List
import numpy as np
import torch
from transformers import TrainingArguments, Trainer
import evaluate
from sklearn.model_selection import train_test_split
import nltk
from transformers import AutoTokenizer
from torch.utils.data import Dataset
nltk.download('treebank')
from transformers import AutoTokenizer, AutoModelForTokenClassification
# load tree bank dataset
tagged_sentences = nltk.corpus.treebank.tagged_sents()
print("Number of samples:", len(tagged_sentences))

# save sentences and tags
sentences, sentence_tags = [], []
for tagged_sentence in tagged_sentences:
    sentence, tags = zip(*tagged_sentence)
    sentences.append([word.lower() for word in sentence])
    sentence_tags.append([tag for tag in tags])


train_sentences, test_sentences, train_tags, test_tags = train_test_split(
    sentences,
    sentence_tags,
    test_size=0.3
)

valid_sentences, test_sentences, valid_tags, test_tags = train_test_split(
    test_sentences,
    test_tags,
    test_size=0.5
)

model_name = "QCRI/bert-base-multilingual-cased-pos-english"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True
)

MAX_LEN = 256

class PosTagging_Dataset(Dataset):
    def __init__(self,
                 sentences: List[List[str]],
                 tags: List[List[str]],
                 tokenizer,
                 label2id,
                 max_len=MAX_LEN):
        super().__init__()
        self.sentences = sentences
        self.tags = tags
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        input_token = self.sentences[idx]
        label_token = self.tags[idx]

        input_token = self.tokenizer.convert_tokens_to_ids(input_token)
        attention_mask = [1] * len(input_token)
        labels = [self.label2id[token] for token in label_token]

        return {
            "input_ids": self.pad_and_truncate(input_token, pad_id=self.tokenizer.pad_token_id),
            "labels": self.pad_and_truncate(labels, pad_id=self.label2id["O"]),
            "attention_mask": self.pad_and_truncate(attention_mask, pad_id=0)
        }

    def pad_and_truncate(self, inputs: List[int], pad_id: int):
        if len(inputs) < self.max_len:
            padded_inputs = inputs + [pad_id] * (self.max_len - len(inputs))
        else:
            padded_inputs = inputs[:self.max_len]
        return torch.as_tensor(padded_inputs)
    
train_dataset = PosTagging_Dataset(train_sentences, train_tags, tokenizer, label2id)
val_dataset = PosTagging_Dataset(valid_sentences, valid_tags, tokenizer, label2id)
test_dataset = PosTagging_Dataset(test_sentences, test_tags, tokenizer, label2id)




model_name = "QCRI/bert-base-multilingual-cased-pos-english"
model = AutoModelForTokenClassification.from_pretrained(model_name)


accuracy = evaluate.load("accuracy")
ignore_label = len(label2id)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mask = labels != ignore_label
    predictions = np.argmax(predictions, axis=-1)
    return accuracy.compute(predictions=predictions[mask], references=labels[mask])


training_args = TrainingArguments(
    output_dir="out_dir",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
# tokenization
test_sentence = "We are exploring the topic of deep learning"
input = torch.as_tensor([tokenizer.convert_tokens_to_ids(test_sentence.split())])
input = input.to("cuda")

# prediction
outputs = model(input)
_, preds = torch.max(outputs.logits, -1)
preds = preds[0].cpu().numpy()

# decode
pred_tags = ""
for pred in preds:
    pred_tags += id2label[pred] + " "
pred_tags # => PRP VBP RB DT NN IN JJ NN
