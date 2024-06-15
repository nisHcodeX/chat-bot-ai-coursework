# import chatbot
from flask import Flask, jsonify, request, send_from_directory
from chatbot import get_response_final
app = Flask(__name__, static_folder='./frontend/')

@app.route('/')

def home():
    return send_from_directory(app.static_folder, 'index.html')

@app.post('/predict')

def predict():
    text = request.get_json().get("message")
    response = get_response_final(text)

    res = {"answer": response}
    res_msg = jsonify(res)
    return res_msg

if __name__ == "__main__" :
    app.run(debug = True)
