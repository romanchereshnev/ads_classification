# ads_classification

## Idea
The final idea for solving the problem: sentences from different files are vectorized using fastText. Then I use the bidirectional RNN to predict ads based on sequences of sentences (sequence labeling).

Because I do not know the full business processes (I do not know which are more crucial: precision or recall) I decided to focus on F1-score.

## Files
* [utils.py](https://github.com/romanchereshnev/ads_classification/blob/master/utils.py) - contains function from notebooks.
* [helpers.py](https://github.com/romanchereshnev/ads_classification/blob/master/helpers.py) - contains several functions that process data.
* [fasttext_classifier.py](https://github.com/romanchereshnev/ads_classification/blob/master/fasttext_classifier.py) - old script that I wrote some time ago. Contains a better interface for fastText.
* [requirements.txt](https://github.com/romanchereshnev/ads_classification/blob/master/requirements.txt) - packages versions.`
* [1. Create data.ipynb](https://github.com/romanchereshnev/ads_classification/blob/master/1.%20Create%20data.ipynb) - jupyter notebook with data preprocessing steps and creating embedding vectors.
* [2. Ads classification.ipynb](https://github.com/romanchereshnev/ads_classification/blob/master/2.%20Ads%20classification.ipynb) - jupyter notebook with the main results of sequence labeling of advertisements.
* [3. Ads classification without stop words.ipynb](https://github.com/romanchereshnev/ads_classification/blob/master/3.%20Ads%20classification%20without%20stop%20words.ipynb) - jupyter notebook with experements without stop words.
* [4. Ads classification with lemmatization and without stop words.ipynb](https://github.com/romanchereshnev/ads_classification/blob/master/4.%20Ads%20classification%20with%20lemmatization%20and%20without%20stop%20words.ipynb) - jupyter notebook with experements without stop words and with lemmatized text.
