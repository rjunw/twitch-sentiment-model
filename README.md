# Twitch Sentiment Model
## Authors: Ryan Jun Wang, Feifei Li, Victor Trinh

Current best weights: https://drive.google.com/drive/folders/14NzJWjnIOebhR5GvZAITDAPBLXlrhC6G?usp=sharing (best from 50eps base BERT cased; ~75.7% validation accuracy using 768-dim Word2Vec summed with BERT `[CLS]` token)

* `predict.py` should be run in the same file structure as the google drive
* `py predict.py bert-base-cased .\\weights\\casedbertbase_w2v768sum_silu_50eps_seed42_labeled w2v_labeled.model "S OMEGALUL BAD"` 

For a sample product using our model, check out this [repo](https://github.com/Victor-Trinh/twitch-sentiment-webapp)
