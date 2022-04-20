# FeelsAmazingMan: Combining static and contextual representations to improve Twitch sentiment analysis
## Authors: Ryan Jun Wang, Feifei Li, Victor Trinh

The `.ipynb` notebooks can be used to train our models on your own machines. The Kobs 21 data set can be found [here](https://github.com/konstantinkobs/emote-controlled/blob/master/data/labeled_dataset.csv).

## Dependencies

```
emoji
pytorch
pandas
transformers
gensim
```

## Current Results

Best weights: https://drive.google.com/drive/folders/14NzJWjnIOebhR5GvZAITDAPBLXlrhC6G?usp=sharing (best from 50eps base BERT cased; ~75.7% validation accuracy using 768-dim Word2Vec summed with BERT `[CLS]` token)

* `predict.py` should be run in the same file structure as the google drive, refer to the instructions in the `main` block to run

For a sample product using our model, check out this [repo](https://github.com/Victor-Trinh/twitch-sentiment-webapp)
