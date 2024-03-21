import pandas as pd
import ast
from typing import List, Dict
from collections import defaultdict

# Read the CSV files
raw_recipes = pd.read_csv('static/data/RAW_recipes_cut.csv')
pp_recipes = pd.read_csv('static/data/PP_recipes_cut.csv')
interactions = pd.read_csv('static/data/RAW_interactions_cut.csv')

# Merge recipes and interactions to extract data
merged_data = pd.merge(raw_recipes, pp_recipes, on='id', how='inner')
recipes = merged_data[['id', 'name', 'minutes', 'ingredient_ids', 'ingredients', 'steps','description']]
recipes_review_merge = pd.merge(recipes, interactions, left_on='id',right_on='recipe_id', how='inner')
recipes_review = recipes_review_merge[['id', 'name', 'minutes', 'ingredient_ids', 'ingredients', 'steps','description', 'rating','review']]
print(recipes_review.head())

# Compute average rating and combine reviews into a list of reviews
# Drop duplicated rows
df = pd.DataFrame(recipes_review)
df['avg_rating']  = df.groupby('id')['rating'].transform('mean')
df_review = df.groupby('id')['review'].agg(list).reset_index()
df = df.drop(['rating', 'review'], axis=1).drop_duplicates()
result = pd.merge(df, df_review, on='id', how='inner')
print(result.head())

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