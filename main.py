# import chatbot
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder='./frontend/')

@app.route('/')

def home():
    return send_from_directory(app.static_folder, 'index.html')
if __name__ == "__main__" :
    app.run(debug = True)
