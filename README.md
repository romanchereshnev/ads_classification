# ads_classification

## Idea
The final idea for solving the problem: sentences from different files are vectorized using fastText. Then I use the bidirectional RNN to predict ads based on sequences of sentences (sequence labeling).

Because I do not know the full business processes (I do not know which are more crucial: precision or recall) I decided to focus on F1-score.

## Files
* main.py - contains entry point for classification. If varibale "load" equal to True, than script load pre-trained model. Set this valiable to False to re-train the model.
* helpers.py - contains several functions that process data.
* fasttext_classifier.py - old script that I wrote some time ago. Contains a better interface for fastText.
* model - pre-trained model.
* sentences.pkl - fastText vectorized sentences. If you delete this file script will create new vectors and save them into sentences.pkl.
