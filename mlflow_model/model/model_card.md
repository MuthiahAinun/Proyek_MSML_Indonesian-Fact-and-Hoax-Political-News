
# Indonesian DistilBERT base model (uncased) 

## Model description
This model is a distilled version of the [Indonesian BERT base model](https://huggingface.co/cahya/bert-base-indonesian-1.5G).
This model is uncased.

This is one of several other language models that have been pre-trained with indonesian datasets. More detail about 
its usage on downstream tasks (text classification, text generation, etc) is available at [Transformer based Indonesian Language Models](https://github.com/cahya-wirawan/indonesian-language-models/tree/master/Transformers)

## Intended uses & limitations

### How to use
You can use this model directly with a pipeline for masked language modeling:
```python
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='cahya/distilbert-base-indonesian')
>>> unmasker("Ayahku sedang bekerja di sawah untuk [MASK] padi")

[
  {
    "sequence": "[CLS] ayahku sedang bekerja di sawah untuk menanam padi [SEP]",
    "score": 0.6853187084197998,
    "token": 12712,
    "token_str": "menanam"
  },
  {
    "sequence": "[CLS] ayahku sedang bekerja di sawah untuk bertani padi [SEP]",
    "score": 0.03739545866847038,
    "token": 15484,
    "token_str": "bertani"
  },
  {
    "sequence": "[CLS] ayahku sedang bekerja di sawah untuk memetik padi [SEP]",
    "score": 0.02742469497025013,
    "token": 30338,
    "token_str": "memetik"
  },
  {
    "sequence": "[CLS] ayahku sedang bekerja di sawah untuk penggilingan padi [SEP]",
    "score": 0.02214187942445278,
    "token": 28252,
    "token_str": "penggilingan"
  },
  {
    "sequence": "[CLS] ayahku sedang bekerja di sawah untuk tanam padi [SEP]",
    "score": 0.0185895636677742,
    "token": 11308,
    "token_str": "tanam"
  }
]

```
Here is how to use this model to get the features of a given text in PyTorch:
```python
from transformers import DistilBertTokenizer, DistilBertModel

model_name='cahya/distilbert-base-indonesian'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)
text = "Silakan diganti dengan text apa saja."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```
and in Tensorflow:
```python
from transformers import DistilBertTokenizer, TFDistilBertModel

model_name='cahya/distilbert-base-indonesian'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = TFDistilBertModel.from_pretrained(model_name)
text = "Silakan diganti dengan text apa saja."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
```

## Training data

This model was distiled with 522MB of indonesian Wikipedia and 1GB of
[indonesian newspapers](https://huggingface.co/datasets/id_newspapers_2018).
The texts are lowercased and tokenized using WordPiece and a vocabulary size of 32,000. The inputs of the model are 
then of the form:

```[CLS] Sentence A [SEP] Sentence B [SEP]```
