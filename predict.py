import sys
import pandas as pd
import torch 
import torch.nn as nn
from transformers import AutoTokenizer, BertModel
from gensim.models import Word2Vec
import os

class BertForSentenceClassification(BertModel):
    def __init__(self, config, use_dropout, embed_size, dropout = 0.5):
        super().__init__(config)
        
        self.dropout = nn.Dropout(p = dropout)
        self.l1 = nn.Linear(config.hidden_size, 128) #nn.Linear(config.hidden_size + embed_size, 128)
        self.a = nn.SiLU()
        self.l2 = nn.Linear(128, config.num_labels)

        self.loss = torch.nn.CrossEntropyLoss()
        self.use_dropout = use_dropout

    def forward(self, labels=None, w2v_embeds = None, **kwargs):
        outputs = super().forward(**kwargs)

        cls_token_repr = outputs.pooler_output
        if w2v_embeds is not None:
          cls_token_repr = nn.functional.normalize(cls_token_repr) + nn.functional.normalize(w2v_embeds) #torch.cat((cls_token_repr, w2v_embeds), dim = -1)

        # apply dropout
        if self.use_dropout:
          dropouts = self.dropout(cls_token_repr)
        else:
          dropouts = cls_token_repr

        linear = self.l1(dropouts)
        activation = self.a(linear)
        logits = self.l2(activation)
        if labels is not None:
            outputs = (logits, self.loss(logits, labels))
        else:
            outputs = (logits,)
        return outputs

def predict(inputs, inputs_w2v, tokenizer, model):
  with torch.no_grad():
    max_len = max([len(tokenizer(datapoint)['input_ids']) for datapoint in inputs])
    tokenized_inputs = tokenizer(
      inputs.tolist(),          # Input text
      add_special_tokens=True,  # add '[CLS]' and '[SEP]'
      padding='max_length',     # pad to a length specified by the max_length
      max_length=max_len,       # truncate all sentences longer than max_length
      return_tensors='pt',      # return everything we need as PyTorch tensors
    )
    output = model(input_ids = tokenized_inputs['input_ids'],
                   w2v_embeds = inputs_w2v)
    output = nn.functional.softmax(output[0], dim = 1).detach().numpy()
    output = output.argmax(axis = 1)
    
  
  return output




#print(f"tokenizer: {tokenizer} \nweights: {model_path} \nmessage: {message}")


def setup_w2v(messages_df): 
  # recap args
  w2v_path = os.path.join(os.getcwd(), "w2v_labeled.model")

  # load and train w2v then get embeds
  w2v_model = Word2Vec.load(w2v_path) # need to recall this after each prediction
  messages_w2v = [message.split(' ') for message in messages_df.msgs] # format messages
  w2v_model.build_vocab(messages_w2v, update = True) # update w2v vocab
  w2v_model.train(messages_w2v, total_examples = len(messages_w2v), epochs = w2v_model.epochs) # train on new messages
  messages_w2v = torch.Tensor([w2v_model.wv[message].sum(axis = 0) for message in messages_w2v]) # get w2v embeds

  return messages_w2v, messages_w2v[-1]

def setup_model(messages_df, embed_size):
  tokenizer = "bert-base-cased"
  model_path = os.path.join(os.getcwd(), "model") 

  # load model
  model = BertForSentenceClassification.from_pretrained(model_path, use_dropout = False, embed_size = embed_size)
  bert_tokenizer = AutoTokenizer.from_pretrained(tokenizer)

  return bert_tokenizer, model

if __name__ == "__main__":
  """
  https://drive.google.com/drive/folders/16v-xz-IHzthTfUhZWmmK664HLRGdiXzE 
  https://drive.google.com/drive/folders/1P6werqtDGQAzlAUuUajNA5MfanNpR8lx 
  """

  messages = ["omegalul"]

  messages_df = pd.DataFrame({'msgs' : messages}) # messages is an array
  messages_w2v, embed_size = setup_w2v(messages_df)
  bert_tokenizer, model = setup_model(messages, embed_size)

  print(predict(messages_df.msgs, messages_w2v, bert_tokenizer, model))
  messages_w2v, embed_size = setup_w2v(messages_df)
