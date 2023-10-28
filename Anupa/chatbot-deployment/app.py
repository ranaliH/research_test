from flask import Flask, render_template, request, jsonify
# from flask_cors import CORS
import sys
sys.path.append('./Anupa/Chatbot')
from Chatbot_001 import *
#The method of my chatbot1 (get response)

app = Flask(__name__)
# CORS(app)

@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # TODO: check if text is valid

    response = all_func(text,False)# (see comment in line 3)
    # response = "hii"
    print(response)
    message ={"answer": response}
    return jsonify(message)
    

if __name__ == "__main__":
    app.run(debug=True)


#to run ---> py app.py      - Run app.py
#.\venv\Scripts\Activate    - activate VM