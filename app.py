from flask import Flask, render_template, request
from flask import jsonify
import requests
from dotenv import load_dotenv
import os
import re
from geo_chem import generate_geochemistry_response
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import json

app = Flask(__name__)


# def deep_convert_np_to_lists(obj):
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     elif isinstance(obj, dict):
#         return {k: deep_convert_np_to_lists(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [deep_convert_np_to_lists(item) for item in obj]
#     elif isinstance(obj, tuple):
#         return tuple(deep_convert_np_to_lists(item) for item in obj)
#     return obj



# def deep_convert_np_to_lists(obj):
    
#     if isinstance(obj, np.ndarray):
#         # Convert NaNs to None in a NumPy array
#         obj = np.where(np.isnan(obj), None, obj)
#         return obj.tolist()
#     elif isinstance(obj, dict):
#         # Recursively apply to dictionary values
#         return {k: deep_convert_np_to_lists(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         # Recursively apply to list items
#         return [deep_convert_np_to_lists(item) for item in obj]
#     elif isinstance(obj, tuple):
#         # Recursively apply to tuple items
#         return tuple(deep_convert_np_to_lists(item) for item in obj)
#     elif obj != obj:
#         # Replace standalone NaNs with None
#         return None
#     return obj

def deep_convert_np_to_lists(obj):
    if isinstance(obj, np.ndarray):
        # Convert NaNs to None in a NumPy array
        
        return [None if np.isnan(x) else x for x in obj.flat] if obj.ndim == 1 else [deep_convert_np_to_lists(row) for row in obj]
        return obj.tolist()
    elif isinstance(obj, dict):
        # Recursively apply to dictionary values
        return {k: deep_convert_np_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively apply to list items
        return [deep_convert_np_to_lists(item) for item in obj]
    elif isinstance(obj, tuple):
        # Recursively apply to tuple items
        return tuple(deep_convert_np_to_lists(item) for item in obj)
    elif isinstance(obj, float) and np.isnan(obj):
        # Replace standalone NaNs with None
        return None
    return obj

        
def generate_response(user_query, topic):
  """
  This function sends a post request to the RAG model server with the user_query and topic.
  """
  # send post request to the rag model flask server with user_query and topic
  # get the response from the rag model server and return it
  load_dotenv()
  rag_server_url = os.getenv('SERVER_URL')  # Get the server URL from the environment variables
  if (rag_server_url is None) or (rag_server_url == ""):
      return "The server URL is not set. Please set it in the environment variables."
  response = requests.post(rag_server_url + '/chatbot', json={"query": user_query, "topic": topic}, verify=False)
  if response:
      response_json = response.json()
      if 'answer' in response_json:
          answer = response_json['answer']
          answer = re.sub(r'\*\*+', "", answer)
          answer = re.sub(r'\*', "🔸", answer)
          answer = re.sub(r'-(?=\s)', "🔸", answer)
          # context_items = response_json.get('context_items', [])
          if (('so I cannot answer this question from the provided context.' in answer) or ('The context does not mention any information' in answer) or ('The context does not' in answer)):
             answer += '\n\n 📔 Note: Sometimes I am unable to answer the question as I am still learning and improving. Please provide more context or rephrase the question.'
          return answer
      else:
          default_answer = "Sorry, I am unable to generate a response at this moment."
          return default_answer
  else:
      error_message = "There was an error connecting to the chatbot. Please try again later."
      return error_message
  # if(user_query == "hello"):
  #    return "Hello! How can I help you today?"
  # elif(user_query.lower() == "what is ngdr?"):
  #    return "National Geoscience Data Repository (NGDR) is a flagship initiative conceptualised by Ministry of Mines as a part of National Mineral Exploration Policy (NMEP), 2016 for hosting all exploration related geoscientific data for dissemination to all the stakeholders so as to expedite, enhance and facilitate the exploration coverage of the country. Geological Survey of India is selected as the nodal agency for the implementation of NGDR. All legacy data of all stakeholders will be brought in to the system through digitization and all the exploration related data has been standardized through MERT (Mineral Exploration Reporting Template) and converted into GIS compatible formats for application of emerging technologies like AI and ML."
  # return "Thanks for your query! I'm still under development and learning to communicate effectively. Stay tuned for future updates!"

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/chatbot")
def home_chatbot():
  return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/get_response_ngdr", methods=["POST"])
def ngdr_geochem_response():
    user_query = request.json["query"]
    threshold = int(request.json["threshold"])
    # threshold = 95
    # send this user_query to the ngdr main function then get the response and return it by jsonify after making it to a dictionary
    
    response = generate_geochemistry_response(user_query, threshold) 
    # response = generate_geochemistry_response(user_query) 
    # new_response = deep_convert_np_to_lists(response)
    # print(response[0])
    new_response = deep_convert_np_to_lists(response)
    # print("RESPONSE:", new_response)
    return jsonify(new_response)
    # data, layout = generate_ngdr_map()
    # return jsonify(data=data, layout=layout)


# @app.route("/get_response", methods=["POST"])
# def get_response():
#   user_query = request.form["query"]
#   # response = model.generate_response(user_query)  # Call your model function
#   response = generate_response(user_query)  # Call your model function

#   return jsonify({"response": response})
@app.route("/get_response_rag", methods=["POST"])
def get_response():
    user_query = request.json["query"]
    topic = request.json["topic"]
    response = generate_response(user_query, topic)
    return jsonify({"response": response})

if __name__ == "__main__":
  app.run(debug=True)
