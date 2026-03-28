import torch
from transformers import BertForTokenClassification

def build_bert_ner_model(pretrained_path='bert-base-chinese', num_labels=49):
    model = BertForTokenClassification.from_pretrained(
        pretrained_path,
        num_labels=num_labels,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )
    return model