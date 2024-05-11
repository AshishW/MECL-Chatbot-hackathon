## Steps for setting up the flask application:

1. clone the repository or download the zip
2. cd into MECL Chatbot
3. Create a virtual environment as per the [Flask documentation](https://flask.palletsprojects.com/en/3.0.x/installation/#create-an-environment).
4. Activate the virtual environment as per the [Flask documentation](https://flask.palletsprojects.com/en/3.0.x/installation/#activate-the-environment) depending on your Operating System.

eg. terminal commands for Windows:
creating virtual environment:`py -3 -m venv .venv`
activating virtual environment:`.\.venv\Scripts\activate`
5. after activating the environment, install the libraries and packages using `pip install -r requirements.txt`
6. To start the Flask application run the command `flask run --debug`


## Project Structure:

/MECL Chatbot/
|-- app.py
|-- geo_chem.py
|-- NGDR_Nagpur.csv
|-- requirements.txt
|-- .venv/
|-- static/
|   |-- style.css
|   |-- bot.png
|   |-- user.png
|   |-- bot.jpeg
|-- templates/
|   |-- index.html
|-- readme.md
