from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
import sklearn

raw_recipes = pd.read_csv('static/data/RAW_recipes_cut.csv')
pp_recipes = pd.read_csv('static/data/PP_recipes_cut.csv')
interactions = pd.read_csv('static/data/RAW_interactions_cut.csv')

recipes_review_merge = pd.merge(raw_recipes, interactions, left_on='id',right_on='recipe_id', how='inner')
recipes_review = recipes_review_merge[['id', 'name', 'minutes', 'tags', 'ingredients', 'steps','description', 'rating','review']]

df = pd.DataFrame(recipes_review)
df['avg_rating']  = df.groupby('id')['rating'].transform('mean')
df_review = df.groupby('id')['review'].agg(list).reset_index()
df = df.drop(['rating', 'review'], axis=1).drop_duplicates()
result = pd.merge(df, df_review, on='id', how='inner')

recipes = []

for index, row in result.iterrows():
  review = row['review']
  review = ''.join(str(x) for x in review)
  recipes.append((row["name"], str(row["tags"]), str(row["tags"])+str(row['name'])+str(row["steps"])+str(row["description"]), review))

vectorizer = TfidfVectorizer(stop_words = ['english', 'time-to-make', 'course', 'cuisine', 'main-ingredient', 'occasion', 'equipment', 'preparation'], max_df = .8, min_df = 1)
td_matrix = vectorizer.fit_transform([x[2] for x in recipes])

from scipy.sparse.linalg import svds
u, s, v_trans = svds(td_matrix, k=100)

docs_compressed, s, words_compressed = svds(td_matrix, k=100)
words_compressed = words_compressed.transpose()

words_compressed_normed = normalize(words_compressed, axis = 1)

td_matrix_np = td_matrix.toarray()
td_matrix_np = normalize(td_matrix_np)

docs_compressed_normed = normalize(docs_compressed)

# TEST USER INPUT 
query = "easy blueberry breakfast"
query_tfidf = vectorizer.transform([query]).toarray()
query_vec = normalize(np.dot(query_tfidf, words_compressed)).squeeze()

def svd_search(query_vec_in, k = 5):
    sims = docs_compressed_normed.dot(query_vec_in)
    asort = np.argsort(-sims)[:k+1]
    return [(i, recipes[i][0],sims[i]) for i in asort[1:]]

for i, proj, sim in svd_search(query_vec):
    print("({}, {}, {:.4f}".format(i, proj, sim))