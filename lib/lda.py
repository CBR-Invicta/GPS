from typing import Tuple
import gensim
import pandas as pd


# https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24


def _print_lda_dictionary_top(dictionary: gensim.corpora.dictionary.Dictionary):

    count = 0
    for key, value in dictionary.iteritems():
        print(f"{key}: {value}")
        count += 1
        if count > 10:
            break


def _print_lda_topics(lda_model: gensim.models.ldamulticore.LdaMulticore):

    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic: {idx}")
        print(topic)


def add_lda_topic_as_feature(
    processed_df: pd.DataFrame,
    train_dictionary: gensim.corpora.Dictionary,
    lda_model: gensim.models.LdaMulticore,
    csv_col: str,
    lda_topic_col: str,
) -> pd.DataFrame:

    processed_lda_data = []
    for _index, processed_row in processed_df.iterrows():
        processed_lda_data += [processed_row[csv_col].split(",")]

    processed_bow_corpus = [train_dictionary.doc2bow(doc) for doc in processed_lda_data]

    lda_topics = []
    for bow_corpus_item in processed_bow_corpus:
        max_score = 0
        max_topic_idx = None
        for topic_idx, score in lda_model[bow_corpus_item]:
            if score > max_score:
                max_score = score
                max_topic_idx = topic_idx

        lda_topics += [max_topic_idx]

    processed_df[lda_topic_col] = lda_topics
    processed_df[lda_topic_col] = processed_df[lda_topic_col].astype("category")

    return processed_df


def add_lda_topic_from_train_as_feature(
    train_df: pd.DataFrame, test_df: pd.DataFrame, csv_col: str, lda_topic_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train_lda_data = []
    for _index, row in train_df.iterrows():
        train_lda_data += [row[csv_col].split(",")]
    # print(train_lda_data[0])

    train_dictionary = gensim.corpora.Dictionary(train_lda_data)
    # print_lda_dictionary_top(train_dictionary)

    train_dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    # print_lda_dictionary_top(train_dictionary)

    train_bow_corpus = [train_dictionary.doc2bow(doc) for doc in train_lda_data]

    lda_model = gensim.models.LdaMulticore(
        train_bow_corpus, num_topics=20, id2word=train_dictionary, passes=10, workers=2
    )
    # print_lda_topics(lda_model)

    train_df = add_lda_topic_as_feature(
        train_df, train_dictionary, lda_model, csv_col, lda_topic_col
    )
    test_df = add_lda_topic_as_feature(
        test_df, train_dictionary, lda_model, csv_col, lda_topic_col
    )

    return train_df, test_df
