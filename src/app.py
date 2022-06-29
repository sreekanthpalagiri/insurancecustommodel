from flask import Flask, request, jsonify
import pandas as pd
from model import pipe


##import pipeline here

app = Flask(__name__)

@app.route('/',methods=['GET'])
def index():
    content = request.json
    df = pd.DataFrame.from_dict(content)
    return jsonify(pipe.fit_predict(df))


if __name__=='__main__':
    app.run()