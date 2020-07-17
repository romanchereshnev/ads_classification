# ads_classification

## Idea
The final idea for solving the problem: sentences from different files are vectorized using fastText. Then I use the bidirectional RNN to predict ads based on sequences of sentences (sequence labeling).

Because I do not know the full business processes (I do not know which are more crucial: precision or recall) I decided to focus on F1-score.

## Files
* `utils.py - contains function from notebooks.`
* `helpers.py - contains several functions that process data.`
* `fasttext_classifier.py - old script that I wrote some time ago. Contains a better interface for fastText.`
* `requirements.txt - packages versions.`
* `1. Create data.ipynb - jupyter notebook with data preprocessing steps and creating embedding vectors.`
* `2. Ads classification.ipynb - jupyter notebook with the main results of sequence labeling of advertisements.`
* `3. Ads classification without stop words.ipynb - jupyter notebook with experements without stop words.`
* `4. Ads classification with lemmatization and without stop words.ipynb - jupyter notebook with experements without stop words and with lemmatized text.`
