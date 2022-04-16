# twitch-sentiment-model

Current best weights: https://drive.google.com/drive/folders/1P6werqtDGQAzlAUuUajNA5MfanNpR8lx?usp=sharing (~75% accuracy using 768-dim Word2Vec summed with BERT `[CLS]` token)

* `predict.py` should be run in the same file structure as the google drive
* `py predict.py bert-base-cased .\\weights\\casedbertbase_w2v768sum_silu_35eps_seed42_labeled w2v_labeled.model "S OMEGALUL BAD"` -> Gives result `Negative`
