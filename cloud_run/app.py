import pandas as pd
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/', methods=['POST'])
def online_main():

    input_data = request.get_json().get('columns') # dictionary
    #input_data = request['columns'] # dictionary

    df = pd.DataFrame.from_dict(input_data)

    with open('final_model_amazon.pickle', 'rb') as rfile:
        model = pickle.load(rfile)

    preds_probas = model.predict(df,  prediction_type='Probability')

    result = {'probability_0': preds_probas[0][0], 'probability_1': preds_probas[0][1]}

    return jsonify(result)


if __name__ == "__main__":
    app.run()
