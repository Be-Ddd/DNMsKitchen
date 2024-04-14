import pandas as pd
import ast
from typing import List, Dict
from collections import defaultdict
import re

# Read the CSV files
raw_recipes = pd.read_csv('static/data/RAW_recipes_cut.csv')
pp_recipes = pd.read_csv('static/data/PP_recipes_cut.csv')
interactions = pd.read_csv('static/data/RAW_interactions_cut.csv')

# Merge recipes and interactions to extract data
merged_data = pd.merge(raw_recipes, pp_recipes, on='id', how='inner')
recipes = merged_data[['id', 'name', 'minutes', 'ingredient_ids', 'ingredients', 'steps','description']]
recipes_review_merge = pd.merge(recipes, interactions, left_on='id',right_on='recipe_id', how='inner')
recipes_review = recipes_review_merge[['id', 'name', 'minutes', 'ingredient_ids', 'ingredients', 'steps','description', 'rating','review']]
print(recipes_review['ingredients'])

# Compute average rating and combine reviews into a list of reviews
# Drop duplicated rows
df = pd.DataFrame(recipes_review)
df['avg_rating']  = df.groupby('id')['rating'].transform('mean')
df_review = df.groupby('id')['review'].agg(list).reset_index()
df = df.drop(['rating', 'review'], axis=1).drop_duplicates()
result = pd.merge(df, df_review, on='id', how='inner')
#print(result.head())

# Convert data frames to json format
result.to_json('processed_data.json', orient='records', lines=True)

# Construct the inverted index
inv_idx_df = result[['id', 'ingredient_ids']]
inv_idx_dict = inv_idx_df.to_dict('records') 

# Returns a dictionary where ingredient id is mapped to a list of recipe ids
def build_inv_idx(recipe_ing: List[dict]) -> dict:
  res = defaultdict(list)
  for recipe_ing_dict in recipe_ing:
    recipe_id = recipe_ing_dict["id"]
    ingredient_ids = ast.literal_eval(recipe_ing_dict["ingredient_ids"])
    for ingredient_id in ingredient_ids:
      res[str(ingredient_id)].append(recipe_id)
  return dict(res)

inv_idx = build_inv_idx(inv_idx_dict)
print(inv_idx['648'])

#def jaccard(ingr_list1, ingr_list2):
#    set1 = set([stemmer.stem(w.lower()) for w in ingr_list1])
#    set2 = set([stemmer.stem(w.lower()) for w in ingr_list2])
#    if len(set.union(set1, set2)) == 0:
#        return 0
#    return len(set.intersection(set1, set2))/len(set.union(set1, set2))

# split the string by space, ignoring commas
# Returns: list of ingredient strings
def tokenize_ingr_list(ingr_input: str) -> List[str]:
   items = re.split(r'\s*|,\s*', ingr_input)

def jaccard_similarity(list1: List[str], list2: List[str]) -> float:
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return intersection / union

#Returns a dictionary of recipe ids mapped to jaccard similarity scores. Jaccard similarity calculated based on query of ingredients and recipe ingredient lists.  
# Parameters: 
#   Input 1 is a list of ingredients from user query.
#   Input 2 is a set of recipe ids. 
def jacc_dict_ing(ingr_list, set):
  jacc_scores_dict = {}
  #order recipes by jaccard similarity
  for recipe_id in set:
    #find recipe row in df
    recipe_ing_list = recipes_review.loc[recipes_review['id'] == recipe_id, 'ingredients'].values[0]
    #find jaccsim(query, recipe ingredient list)
    jacc_score = jaccard_similarity(ingr_list, recipe_ing_list)
    jacc_scores_dict[recipe_id]= jacc_score
  return jacc_scores_dict