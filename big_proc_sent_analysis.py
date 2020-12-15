# importing packages
import pandas as pd
import numpy as np
from multiprocessing import Pool
from textblob import TextBlob
pd.set_option('display.max_colwidth', -1) # so we can view our entire review

# helper function to parallelize tasks
def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def clean_reviews(df):
    df['text'] = df['text'].replace("\n", " ", regex=True)
    return df

def sentiment_analysis(df):
    df['review_polarity'], df['review_subjectivity'] = TextBlob(df['text'].to_string()).sentiment
    return df

# loading the data
reviews = pd.read_json('data/yelp_academic_dataset_review.json', lines=True)

n_splits = 100
reviews_sep = np.array_split(reviews, n_splits)
for n in range(n_splits):
    print(f'Processing {n} split...')

    this_review = reviews_sep[n]
    # data cleaning
    this_review = parallelize_dataframe(this_review, clean_reviews, n_cores=100)
    # performing sentiment analysis on the data
    this_review = parallelize_dataframe(this_review, sentiment_analysis, n_cores=100)
    this_review.to_csv('data/sent_segs/reviews/yelp_data_prep_' + str(n) + '.csv')
