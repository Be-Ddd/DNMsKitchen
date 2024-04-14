import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import preprocessing

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
LOCAL_MYSQL_USER = "root"
LOCAL_MYSQL_USER_PASSWORD = "Midtv1929"
LOCAL_MYSQL_PORT = 3306
LOCAL_MYSQL_DATABASE = "test"

mysql_engine = MySQLDatabaseHandler(LOCAL_MYSQL_USER,LOCAL_MYSQL_USER_PASSWORD,LOCAL_MYSQL_PORT,LOCAL_MYSQL_DATABASE)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

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
    return sql_search(ingr, mins)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)