# FeelsAmazingMan: Combining static and contextual representations to improve Twitch sentiment analysis
## Authors: Ryan Jun Wang, Feifei Li, Victor Trinh

The `.ipynb` notebooks can be used to train our models on your own machines. The Kobs 20 data set can be found [here](https://github.com/konstantinkobs/emote-controlled/blob/master/data/labeled_dataset.csv); the emote augmentation uses the `emote-average.tsv` file found [here](https://github.com/konstantinkobs/emote-controlled/blob/master/lexica/emote_average.tsv). Download and put these into a data folder in the same directory as the notebooks.

## Dependencies

```
emoji
pytorch
pandas
transformers
gensim
```

## Current Results

 | Model | Accuracy | Macro F-1 |
 | ----- | -------- | --------- |
 | Kobs 20 Sentence CNN | 63.8% | 62.6% |
 | Dolin 21 Bi-gram Random Forest | 71.2% | N/A |
 | Fully-connected MLP + Labeled-Word2Vec (Batch Size 64)  | 60.3% | 45.2% |
 | BERT (Batch Size 64) | 73.9% | 68.1% |
 | BERT + Labeled-Word2Vec (Batch Size 64) | **75.7**% | **68.7**% |
 | BERT + Labeled-Word2Vec + GRU (Batch Size 64, Weight decay 0.1) | (**more stable**) **74.0**% | **68.0**%  |

Best weights: https://drive.google.com/drive/folders/14NzJWjnIOebhR5GvZAITDAPBLXlrhC6G?usp=sharing (best from 50eps base BERT cased; ~75.7% validation accuracy using 768-dim Word2Vec summed with BERT `[CLS]` token)

* `predict.py` should be run in the same file structure as the google drive, refer to the instructions in the `main` block to run

For a sample product using our model, check out this [repo](https://github.com/Victor-Trinh/twitch-sentiment-webapp)

## Contact
You can contact us at ryanjun.wang[at]mail.utoronto.ca, ff.li[at]mail.utoronto.ca, and v.trinh[at]mail.utoronto.ca
