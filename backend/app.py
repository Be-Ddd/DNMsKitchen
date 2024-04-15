import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import preprocessing
import pandas as pd

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'processed_data.json')

# Assuming your JSON data is stored in a file named 'processed_data.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    df = pd.DataFrame(data)
    # episodes_df = pd.DataFrame(data['episodes'])
    # reviews_df = pd.DataFrame(data['reviews'])

app = Flask(__name__)
CORS(app)

def json_search(ingr, mins):
    matches = []
    
    #extract list of ingredients from user query
    ingr_list = preprocessing.tokenize_ingr_list(ingr)
    print("INGREDIENT LIST")
    print(ingr_list)

    #Create a dictionary that maps recipe id to jaccard sim score 
    #Calculate jacc sim score between recipe ing list and query ing list for recipes that contain 1 or more query ingredients. 
    scores = {}
    for index, row in df.iterrows():
        if any(ingredient in row['ingredients'] for ingredient in ingr_list):
            sim_score = preprocessing.jaccard_similarity(ingr_list, row['ingredients'])
            scores[row["id"]] = sim_score
    
    #sort dictionary by sim scores in descending order
    scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    print("SCORES")
    print(scores)

    #Create a pandas df that contains matched recipes. Filter out matched recipes that have cooking time > user inputs max cooking time.  
    matches = df[(df['id'].isin(scores.keys())) & (df['minutes'] <= int(mins))]
    print("MATCHES")
    print(matches)
    
    #Map similarity scores from scores dict to corresponding recipe id in matches df
    matches['similarity_score'] = matches['id'].map(scores)
    print("MATCHES ADDING SIM SCORE")
    print(matches)
    
    #Sort recipes in matches df by similarity score and convert it to JSON format. 
    matches = matches.sort_values(by='similarity_score', ascending=False)
    matches_json = matches.to_json(orient='records')
    print("MATCHES AS JSON")
    print(matches_json)
    return matches_json

    # #if no ingredients in query, then possible outputted recipes = whole recipe set, otherwise filter down recipes based on ingredients in query
    # #find intersection between all recipe_ids that contain inputted ingredients in their ingredient lists (intersection = recipes that contain all of ingredients in query)
    # #jacc_scores_dict is a dictionary with keys = recipe id and values = jacc scores based  on similarity of ingr list to user query of ingredients. 
    # if len(ingr_list) != 0: 
    #     intersection_acc = set(ingr_list[0])
    #     for ing in ingr_list: 
    #         recipes_with_ing = set(preprocessing.inv_idx[ing])
    #         intersection_acc = intersection_acc.intersection(recipes_with_ing) 
        
    #     jacc_scores_dict = preprocessing.jacc_dict_ing(ingr_list,intersection_acc)
        
    #     #if intersection empty, find union of recipes with inputted ingredients (union = recipes that contain 1 or more of ingredients in query)
    #     union_acc = set()
    #     for ing in ingr_list: 
    #         recipes_with_ing = set(preprocessing.inv_idx[ing])
    #         intersection_acc = intersection_acc.union(recipes_with_ing) 
        
    #     jacc_scores_dict = preprocessing.jacc_dict_ing(ingr_list, union_acc)

    #     print("jacc: ", jacc_scores_dict)

# Sample search, the LIKE operator in this case is hard-coded, 
# but if you decide to use SQLAlchemy ORM framework, 
# there's a much better and cleaner way to do this
def sql_search_og(episode):
    query_sql = f"""SELECT * FROM episodes WHERE LOWER( title ) LIKE '%%{episode.lower()}%%' limit 10"""
    keys = ["id","title","descr"]
    data = mysql_engine.query_selector(query_sql)
    return json.dumps([dict(zip(keys,i)) for i in data])

def sql_search(ingr, mins):
    query_sql = f"""SELECT * FROM recipes WHERE ingredients LIKE '%%{ingr}%%' AND minutes <= {mins} limit 10"""
    
    #extract list of ingredients from user query
    ingr_list = preprocessing.tokenize_ingr_list(ingr)

    #if no ingredients in query, then possible outputted recipes = whole recipe set, otherwise filter down recipes based on ingredients in query
    #find intersection between all recipe_ids that contain inputted ingredients in their ingredient lists (intersection = recipes that contain all of ingredients in query)
    #jacc_scores_dict is a dictionary with keys = recipe id and values = jacc scores based  on similarity of ingr list to user query of ingredients. 
    if len(ingr_list) != 0: 
        intersection_acc = set(ingr_list[0])
        for ing in ingr_list: 
            recipes_with_ing = set(preprocessing.inv_idx[ing])
            intersection_acc = intersection_acc.intersection(recipes_with_ing) 
        
        jacc_scores_dict = preprocessing.jacc_dict_ing(ingr_list,intersection_acc)
        
        #if intersection empty, find union of recipes with inputted ingredients (union = recipes that contain 1 or more of ingredients in query)
        union_acc = set()
        for ing in ingr_list: 
            recipes_with_ing = set(preprocessing.inv_idx[ing])
            intersection_acc = intersection_acc.union(recipes_with_ing) 
        
        jacc_scores_dict = preprocessing.jacc_dict_ing(ingr_list, union_acc)
    



    keys = ["id","nam","minutes","ingredients", "steps", "descr", "avgrating", "reviews"]
    data = mysql_engine.query_selector(query_sql)
    return json.dumps([dict(zip(keys,i)) for i in data])

def sql_search_fail(ingr, mins):
    # Split the ingredients into individual items
    ingredients = [ingredient.strip() for ingredient in ingr.split(",")]
    # Construct the SQL query to count matching ingredients and filter recipes
    query_sql = f"""
        SELECT 
            *, 
            (SELECT COUNT(*) FROM ingredients WHERE recipes.id = ingredients.recipe_id AND ingredients.ingredient IN ({', '.join(['%s'] * len(ingredients))})) AS matching_count
        FROM 
            recipes 
        WHERE 
            (SELECT COUNT(*) FROM ingredients WHERE recipes.id = ingredients.recipe_id AND ingredients.ingredient IN ({', '.join(['%s'] * len(ingredients))})) = %s 
            AND minutes <= %s 
        LIMIT 10
    """
    # Parameters for the SQL query: list of ingredients, count of ingredients, and maximum cooking time
    # params = ingredients + [len(ingredients), mins]
    keys = ["id", "nam", "minutes", "ingredients", "steps", "descr", "avgrating", "reviews"]
    # Assuming mysql_engine.query_selector() executes the query with parameters and returns the result
    #data = mysql_engine.query_selector(query_sql, params)
    # Convert the result into JSON format
    data = mysql_engine.query_selector(query_sql)
    return json.dumps([dict(zip(keys, i)) for i in data])


#preprocess csv files and output recipes in JSON format and construct an inverted index mapping ingredient ids to recipe ids. 
recipes = preprocessing.result
inv_idx = preprocessing.inv_idx


@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/recipes")
def recipes_search():
    ingr = request.args.get("ingredient")
    mins = request.args.get("minutes")
    return json_search(ingr, mins)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)