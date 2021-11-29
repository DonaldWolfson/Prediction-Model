# CSE 158 Assignment 2

A linear regression model aimed at finding features that best influence predicting a Reddit post's upvotes.

## Description

This repo stores a research project for investigating the influence of different features in a linear regression model that aims to predict a Reddit posts upvotes. The finalized paper can be found in [pdf/Research_Paper.pdf](pdfs/Research_Paper.pdf).

To start, this project web mined the [top 500 subreddits](scripts/top_subreddits.ipynb) based on subsriber count, and then attempts to get there [top 500 posts](scripts/top_posts.ipynb) from the last 365 days. The top 500 subreddits can be found [here](data/top_subreddits.json), and the 246,472 posts can be found [here](data/top_posts.csv.gz).

The finalized prediction model script can be found [here](scripts/prediction_model.ipynb). Numerous scripts were used to optimize, and analyze the script with ablation. This can be found in the [scripts](scripts) folder. Any images produced by the scripts are stored in the [images](images) folder.

This project was a class assignment for Fall 2021, CSE 158, and the assignment description can be found [here](pdfs/Assignment_2.pdf). The approachs and model was inspired by a research paper that can be found [here](pdfs/Research_Paper.pdf), and the dataset this papers utilizes can be found [here](data/submissions.csv.gz).

## Getting Started

### Dependencies

* [Python](https://www.python.org)
* [NumPy](https://numpy.org)
* [scikit-learn](https://scikit-learn.org/stable/)
* [Matplotlib](https://matplotlib.org)
* [PRAW](https://praw.readthedocs.io/en/stable/#getting-starteds)
* [wordcloud](https://pypi.org/project/wordcloud/)
* [nltk](https://www.nltk.org/install.html)

## Authors

* [@DonaldWolfson](https://github.com/DonaldWolfson)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* [frontpagemetrics.com](https://frontpagemetrics.com/list-all-subreddits)
  * Data Used: 2021-11-19.csv
* [CSE 158 Datasets](https://cseweb.ucsd.edu/~jmcauley/datasets.html#reddit)
  * Understanding the interplay between titles, content, and communities in social media
Himabindu Lakkaraju, Julian McAuley, Jure Leskovec
ICWSM, 2013
  * Data Used: submissions.csv.gz
