import pandas as pd
import ast
from typing import List, Dict
from collections import defaultdict
import re
import helper_functions

def create_df():
  # Read the CSV files
  raw_recipes = pd.read_csv('static/data/RAW_recipes_cut.csv')
  pp_recipes = pd.read_csv('static/data/PP_recipes_cut.csv')
  interactions = pd.read_csv('static/data/RAW_interactions_cut.csv')

  # Merge recipes and interactions to extract data
  merged_data = pd.merge(raw_recipes, pp_recipes, on='id', how='inner')
  recipes = merged_data[['id', 'name', 'minutes', 'ingredient_ids', 'ingredients', 'steps','description', 'calorie_level', 'tags']]
  recipes_review_merge = pd.merge(recipes, interactions, left_on='id',right_on='recipe_id', how='inner')
  recipes_review = recipes_review_merge[['id', 'name', 'minutes', 'ingredient_ids', 'ingredients', 'steps','description', 'rating','review', 'calorie_level', 'tags']]

  # Compute average rating and combine reviews into a list of reviews
  # Drop duplicated rows
  df = pd.DataFrame(recipes_review)
  df['avg_rating']  = df.groupby('id')['rating'].transform('mean')
  df_review = df.groupby('id')['review'].agg(list).reset_index()
  df = df.drop(['rating', 'review'], axis=1).drop_duplicates()
  df = pd.merge(df, df_review, on='id', how='inner')
  df['ingredient_ids'] = df['ingredient_ids'].apply(ast.literal_eval)
  df['ingredients'] = df['ingredients'].apply(ast.literal_eval)
  df['steps'] = df['steps'].apply(ast.literal_eval)

  # remove 0 minute recipes
  df = df[df['minutes']!=0]

  #add columns in dataset df for each dietary restriction
  #categorize food groups
  meat_no_fish = ['chicken','bacon','turkey','beef','pork','duck','turkey', 'steak', 'wings', 'boneless skinless chicken breast halves' ]
  fish = ['fish', 'salmon', 'sardines','trout', 'mackerel', 'cod', 'haddock', 'pollock'
          'flounder', 'tilapia', 'shellfish', 'mussels', 'scallops', 'squid', 'mussels', 
           'oysters', 'crab', 'shrimp', 'sea bass', 'halibut', 'tuna']
  dairy = ['milk', 'ice cream', 'cheese', 'yogurt', 'cream', 'butter', 'buttermilk', 'heavy cream']
  gluten_food = ['bread', 'beer', 'cake', 'pie', 'candy', 'cereal', 'cookie', 'croutons', 'french fries',
                 'gravy', 'seafood', 'malt', 'pasta', 'hot dog', 'salad dressing', 'soy sauce', 'rice seasoning', 
                 'chips', 'chicken', 'soup']
  non_kosher = ['shellfish', 'crab', 'shrimp', 'lobster', 'pork ']
  
  #make dictionary with key = dietary restriction, value = list of ingredients that are NOT allowed
  restriction_dict = {'vegetarian' : (meat_no_fish + fish),
                      'pescatarian': meat_no_fish, 
                      'vegan' : (meat_no_fish + fish + dairy + ['eggs', 'egg']),
                      'lactose-intolerant' : dairy, 
                      'gluten free' : gluten_food, 
                      'peanut-allergy' : ['nuts', 'peanut', 'peanut butter', 'crushed peanuts'], 
                      'kosher' : non_kosher, 
                      'halal' : meat_no_fish }
  
  #add empty col for each dietary restriction to df
  for dietary_restriction in restriction_dict:
    #cells are initialized to ""
    df[dietary_restriction] ='' 

  #label each recipe as satisfying or not satisfying each dietary restriction
  helper_functions.label_recipe_for_all_dietary_restrictions(df, restriction_dict)

  df.to_json('processed_data.json', orient='records')
  return df

result = create_df()
# Construct the inverted index
inv_idx_df = result[['id', 'ingredients']]
inv_idx_dict = inv_idx_df.to_dict('records') 

# Returns a dictionary where ingredient id is mapped to a list of recipe ids
def build_inv_idx(recipe_ing: List[dict]) -> dict:
  res = defaultdict(list)
  for recipe_ing_dict in recipe_ing:
    recipe_id = recipe_ing_dict["id"]
    ingredient_ids = recipe_ing_dict["ingredients"]
    for ingredient_id in ingredient_ids:
      res[str(ingredient_id)].append(recipe_id)
  return dict(res)

inv_idx = build_inv_idx(inv_idx_dict)
# print(inv_idx['648'])

#def jaccard(ingr_list1, ingr_list2):
#    set1 = set([stemmer.stem(w.lower()) for w in ingr_list1])
#    set2 = set([stemmer.stem(w.lower()) for w in ingr_list2])
#    if len(set.union(set1, set2)) == 0:
#        return 0
#    return len(set.intersection(set1, set2))/len(set.union(set1, set2))

# split the string by space, ignoring commas
# Returns: list of ingredient strings
def tokenize_ingr_list(ingr_input: str) -> List[str]:
  print("ingr_input", ingr_input)
  items = re.split(r',\s*', ingr_input)
  # tokens = []
  # for item in items:
  #   tokens.extend(item.split())
  return items

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
def jacc_dict_ing(ingr_list, input_set):
  jacc_scores_dict = {}
  #order recipes by jaccard similarity
  for recipe_id in input_set:
    #find recipe row in df
    recipe_ing_list = recipes_review.loc[recipes_review['id'] == recipe_id, 'ingredients'].values[0]
    #find jaccsim(query, recipe ingredient list)
    jacc_score = jaccard_similarity(ingr_list, recipe_ing_list)
    print("score",jacc_score)
    jacc_scores_dict[recipe_id]= jacc_score
  return jacc_scores_dict

"""Returns true if the recipe ingredient list does not contain any ingredients in the avoid ingredient list. Return false otherwise."""
def avoid_ingr(avoid_lst, recipe):
  # for ingr in recipe["ingredients"]:
  #   if any(avoid_item in ingr for avoid_item in avoid_lst):
  #     return False
  # return True
  recipe_ing_set = set(recipe['ingredients']) 
  avoid_set= set(avoid_lst)
  if len(recipe_ing_set.intersection(avoid_set))==0:
    return True     
  else:
    return False


def insertion_cost(message, j):
    return 1


def deletion_cost(query, i):
    return 1


def substitution_cost(query, message, i, j):
    if query[i - 1] == message[j - 1]:
        return 0
    else:
        return 1

def edit_matrix(query, message, ins_cost_func, del_cost_func, sub_cost_func):
    """Calculates the edit matrix

    Arguments
    =========

    query: query string,

    message: message string,

    ins_cost_func: function that returns the cost of inserting a letter,

    del_cost_func: function that returns the cost of deleting a letter,

    sub_cost_func: function that returns the cost of substituting a letter,

    Returns:
        edit matrix {(i,j): int}
    """

    m = len(query) + 1
    n = len(message) + 1

    chart = {(0, 0): 0}
    for i in range(1, m):
        chart[i, 0] = chart[i - 1, 0] + del_cost_func(query, i)
    for j in range(1, n):
        chart[0, j] = chart[0, j - 1] + ins_cost_func(message, j)
    for i in range(1, m):
        for j in range(1, n):
            chart[i, j] = min(
                chart[i - 1, j] + del_cost_func(query, i),
                chart[i, j - 1] + ins_cost_func(message, j),
                chart[i - 1, j - 1] + sub_cost_func(query, message, i, j),
            )
    return chart

def edit_distance(
    query: str, message: str, ins_cost_func: int, del_cost_func: int, sub_cost_func: int
) -> int:
    """Finds the edit distance between a query and a message using the edit matrix

    Arguments
    =========
    query: query string,

    message: message string,

    ins_cost_func: function that returns the cost of inserting a letter,

    del_cost_func: function that returns the cost of deleting a letter,

    sub_cost_func: function that returns the cost of substituting a letter,

    Returns:
        edit cost (int)
    """

    query = query.lower()
    message = message.lower()
    D= edit_matrix(query,message,ins_cost_func,del_cost_func,sub_cost_func)
    return (D[len(query),len(message)])




def edit_distance_search(
    query: str,
    msgs: List[str],
) -> str:
    """Edit distance search

    Arguments
    =========
    query: string,
        The query we are looking for.

    msgs: list of strings

    ins_cost_func: function that returns the cost of inserting a letter,

    del_cost_func: function that returns the cost of deleting a letter,

    sub_cost_func: function that returns the cost of substituting a letter,

    Returns
    =======
    result: the closest message to the query

    """
    lst=[]
    for msg in msgs:
      dis=edit_distance(query,msg,insertion_cost,deletion_cost,substitution_cost)
      lst.append((dis,msg))
    return sorted(lst, key=lambda tup: tup[0])[0][1]
                    