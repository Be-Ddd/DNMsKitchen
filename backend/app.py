from __future__ import print_function
import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import preprocessing
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
from textblob import TextBlob


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
    print("DFFFFFFF", df)
    # episodes_df = pd.DataFrame(data['episodes'])
    # reviews_df = pd.DataFrame(data['reviews'])

app = Flask(__name__)
CORS(app)

# ingr is the ingredients input, mins is the minutes input, svd is the freeforms
def json_search(ingr, mins, svd, avoid, calorie, selected_diets, appliances):
    matches = []
    
    #extract list of ingredients and appliances from user query
    ingr_list = preprocessing.tokenize_ingr_list(ingr)
    #print(ingr_list)
    ingrs = list(preprocessing.inv_idx.keys())
    for ingr in range(len(ingr_list)):
        #if ingredient not in list of ingredients, find closest ingredient in list
        if ingr_list[ingr] not in preprocessing.inv_idx:
            ingr_list[ingr] = preprocessing.edit_distance_search(ingr_list[ingr],ingrs )
    #print("input_list", ingr_list)
    # print("TYPE OF AVOID")
    # print(avoid)
    # if (avoid == None):
    #     print("NONEEEEE")
    avoid_list = preprocessing.tokenize_ingr_list(avoid)
    if avoid!= "":
        for avoid in range(len(avoid_list)):
            #if ingredient not in list of ingredients, find closest ingredient in list
            if avoid_list[avoid] not in preprocessing.inv_idx:
                avoid_list[avoid] = preprocessing.edit_distance_search(avoid_list[avoid], list(preprocessing.inv_idx.keys()))
        print("avoid_list", avoid_list)
    
    #if no diets selected, then selected_diets = "". Only process selected_diets input if its != "".
    if selected_diets != "": 
        #extract list of dietary restrictions from user query
        selected_diets = preprocessing.tokenize_ingr_list(selected_diets)
        #make letters all lowercase for dietary restrictions
        selected_diets = [dietary_restriction.lower() for dietary_restriction in selected_diets]

    #if no appliances selected, then appliances = "". Only process appliances input if its != "".
    if appliances != "": 
        #extract list of dietary restrictions from user query
        appliances = preprocessing.tokenize_ingr_list(appliances)
        #make letters all lowercase for dietary restrictions
        appliances = [appliance.lower() for appliance in appliances]

    #Create a dictionary that maps recipe id to jaccard sim score 
    #Calculate jacc sim score between recipe ing list and query ing list for recipes that contain 1 or more query ingredients. 
    scores = {}
    for index, row in df.iterrows():
        if any(ingredient in row['ingredients'] for ingredient in ingr_list):
            sim_score = preprocessing.jaccard_similarity(ingr_list, row['ingredients'])
            if preprocessing.avoid_ingr(avoid_list, row):
                scores[row["id"]] = sim_score
    
    #sort dictionary by sim scores in descending order
    #scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    scores = {key: round(value, 3) for key, value in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    print("SCORES")
    print(scores)

    if (int(calorie) <= 100):
        calorie_level = 0
    elif (100 < int(calorie) <= 400):
        calorie_level = 1
    else: 
        calorie_level = 2

    #Create a pandas df that contains matched recipes. Filter out matched recipes that have cooking time > user inputs max cooking time.  
    matches = df[(df['id'].isin(scores.keys())) & (df['minutes'] <= int(mins)) & (df['calorie_level'] <= int(calorie_level))]
    print("MATCHES")
    print(matches) 
    

    #Map similarity scores from scores dict to corresponding recipe id in matches df
    matches['jacc_sim'] = matches['id'].map(scores)
    print("LENGTH OF LIST SELECTED DIETS")
    print(selected_diets)
    print("LENGTHH")
    print(str((len(selected_diets))))
    print("MATCHES ADDING SIM SCORE")
    print(matches)
  
    #Filter out matched recipes that do not satisfy at least 1 of the user's selected dietary restrictions (take union of dietary restriction recipes).
    #if no dietary restrictions to filter for, skip this part. 
    if (len(selected_diets)!=0):
        print("COLUMNS: " + matches.columns.values)
        print("FIND DF SUBSETS WHICH SATISFY DIETARY RESTRICTIONS")
        diet_dfs = [] #holds list of df subsets 
        for dietary_restriction in selected_diets: 
            print("DIETARY RESTRICTION:" + str(dietary_restriction))

        for dietary_restriction in selected_diets: 
            print("DIETARY RESTRICTION:" + str(dietary_restriction))
            new_df = matches.loc[matches[dietary_restriction]=='Y']
            print("NEW DF")
            print(new_df)
            diet_dfs.append(new_df)
            print("UPDATED LENGTH OF LIST DIET_DFS:" + str(len(diet_dfs)))
        
        #merge df subsets (take union of recipes that adhere to at least 1 dietary restriction)
        matches = pd.concat(diet_dfs)
        print("CONCAT")
        print(matches)
        print("DROP DUPS")
        matches = matches.drop_duplicates(subset=["id"])
        print(matches)

        print("MERGED DIET DFS")
        print(matches)


    
    
    # # Add the appliance_score column to the matches dataframe
    # print("APPLIANCE SCORES")
    # print(matches)


    #new
    query_tfidf = vectorizer.transform([svd]).toarray()
    query_vec = normalize(np.dot(query_tfidf, words_compressed)).squeeze()
    svd_scores = svd_search(query_vec)
    print("SVD SCORES")
    print(svd_scores)

    matches['svd_sim'] = matches['id'].map(svd_scores)
    matches['svd_sim'] = matches['svd_sim'].round(3)


    #sentiment analysis
    matches['sentiment'] = matches['review'].apply(get_sentiment)
    matches['sentiment'] = matches['sentiment'].round(3)

    matches['sentiment_category'] = matches['sentiment'].apply(get_sentiment_text)
    
    #appliance input handling
    matches['appliance_score'] = 0

    for index, row in matches.iterrows():
        appliance_count = 0
        for appliance in appliances:
            # Check if the appliance is found in either the "tags" or "steps" columns of the recipe
            if (appliance in row['tags']):
                appliance_count += 1
            for step in row['steps']:
                if appliance in step:
                    appliance_count += 1
                    continue
        # Calculate the percentage of appliances that match
        appliance_score = (appliance_count*1.0 / len(appliances))
        matches.at[index, 'appliance_score'] = appliance_score

        
    #Sort recipes in matches df by similarity score and convert it to JSON format. 
    matches = matches.sort_values(by=['appliance_score', 'jacc_sim', 'svd_sim', 'sentiment'], ascending=False)
    matches_json = matches.to_json(orient='records')
    print("MATCHES AS JSON WITH APPLIANCES")
    print(matches)
    return matches_json


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
    svd = request.args.get("svd", default=" ")
    avoid = request.args.get("avoid")
    calorie = request.args.get("calorie", default="10000")
    selected_diets = request.args.get("diet")
    appliances = request.args.get("appliances")
    response = json_search(ingr, mins, svd, avoid, calorie, selected_diets, appliances)
    return response

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)


# sentiment analysis
def get_sentiment(reviews):
    combined_reviews = ' '.join(reviews)
    analysis = TextBlob(combined_reviews)
    return analysis.sentiment[0]

def get_sentiment_text(sent):
    if sent < -0.5:
        return "Very Negative"
    elif -0.5 <= sent < -0.1:
        return "Fairly Negative"
    elif -0.1 <= sent <= 0.1:
        return "Neutral"
    elif 0.1 < sent <= 0.5:
        return "Fairly Positive"
    else:
        return "Very Positive"

#SVD 

raw_recipes = pd.read_csv('static/data/RAW_recipes_cut.csv')
pp_recipes = pd.read_csv('static/data/PP_recipes_cut.csv')
interactions = pd.read_csv('static/data/RAW_interactions_cut.csv')

recipes_review_merge = pd.merge(raw_recipes, interactions, left_on='id',right_on='recipe_id', how='inner')
recipes_review = recipes_review_merge[['id', 'name', 'minutes', 'tags', 'ingredients', 'steps','description', 'rating','review']]

df2 = pd.DataFrame(recipes_review)
df2['avg_rating']  = df2.groupby('id')['rating'].transform('mean')
df_review = df2.groupby('id')['review'].agg(list).reset_index()
df2 = df2.drop(['rating', 'review'], axis=1).drop_duplicates()
result = pd.merge(df2, df_review, on='id', how='inner')

recipes = []

for index, row in result.iterrows():
  review = row['review']
  review = ''.join(str(x) for x in review)
  recipes.append((row["name"], str(row["tags"]), str(row['name'])+str(row["description"])+str(row["review"]), review))

vectorizer = TfidfVectorizer(stop_words = ['english', 'time-to-make', 'course', 'cuisine', 'main-ingredient', 'occasion', 'equipment', 'preparation'], max_df = .8, min_df = 1)
td_matrix = vectorizer.fit_transform([x[2] for x in recipes])

from scipy.sparse.linalg import svds
u, s, v_trans = svds(td_matrix, k=100)

docs_compressed, s, words_compressed = svds(td_matrix, k=40)
words_compressed = words_compressed.transpose()

words_compressed_normed = normalize(words_compressed, axis = 1)

td_matrix_np = td_matrix.toarray()
td_matrix_np = normalize(td_matrix_np)

docs_compressed_normed = normalize(docs_compressed)

# TEST USER INPUT 
query = "easy blueberry breakfast"
query_tfidf = vectorizer.transform([query]).toarray()
query_vec = normalize(np.dot(query_tfidf, words_compressed)).squeeze()

def svd_search(query_vec_in):
  sims = docs_compressed_normed.dot(query_vec_in)
  asort = np.argsort(-sims)
  results_dict = {}
  for i in asort[1:]:
    results_dict[i] = sims[i]
  return results_dict
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
# def sql_search_og(episode):
#     query_sql = f"""SELECT * FROM episodes WHERE LOWER( title ) LIKE '%%{episode.lower()}%%' limit 10"""
#     keys = ["id","title","descr"]
#     data = mysql_engine.query_selector(query_sql)
#     return json.dumps([dict(zip(keys,i)) for i in data])

# def sql_search(ingr, mins):
#     query_sql = f"""SELECT * FROM recipes WHERE ingredients LIKE '%%{ingr}%%' AND minutes <= {mins} limit 10"""
    
#     #extract list of ingredients from user query
#     ingr_list = preprocessing.tokenize_ingr_list(ingr)

#     #if no ingredients in query, then possible outputted recipes = whole recipe set, otherwise filter down recipes based on ingredients in query
#     #find intersection between all recipe_ids that contain inputted ingredients in their ingredient lists (intersection = recipes that contain all of ingredients in query)
#     #jacc_scores_dict is a dictionary with keys = recipe id and values = jacc scores based  on similarity of ingr list to user query of ingredients. 
#     if len(ingr_list) != 0: 
#         intersection_acc = set(ingr_list[0])
#         for ing in ingr_list: 
#             recipes_with_ing = set(preprocessing.inv_idx[ing])
#             intersection_acc = intersection_acc.intersection(recipes_with_ing) 
        
#         jacc_scores_dict = preprocessing.jacc_dict_ing(ingr_list,intersection_acc)
        
#         #if intersection empty, find union of recipes with inputted ingredients (union = recipes that contain 1 or more of ingredients in query)
#         union_acc = set()
#         for ing in ingr_list: 
#             recipes_with_ing = set(preprocessing.inv_idx[ing])
#             intersection_acc = intersection_acc.union(recipes_with_ing) 
        
#         jacc_scores_dict = preprocessing.jacc_dict_ing(ingr_list, union_acc)
    

#     keys = ["id","nam","minutes","ingredients", "steps", "descr", "avgrating", "reviews"]
#     data = mysql_engine.query_selector(query_sql)
#     return json.dumps([dict(zip(keys,i)) for i in data])
