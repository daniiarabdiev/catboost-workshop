import gcsfs
import pandas as pd
import json
import pickle
from flask import jsonify
from catboost import CatBoost


def read_model(path_model, fs):
    with fs.open(path_model, 'rb') as f:
        model = pickle.load(f)
    return model


def online_main(request):
    # STATIC DATA
    the_project = 'autoinsight-258217'
    with open('catboostworkshop-e91a753d9550.json', 'rb') as rfile:
        token_dic = json.load(rfile)

    bucket = 'catboost-workshop'

    #input_data = request.get_json().get('columns') # dictionary
    input_data = request['columns'] # dictionary

    df = pd.DataFrame.from_dict(input_data)
    print(df)
    # READ METADATA
    fs = gcsfs.GCSFileSystem(project=the_project, token=token_dic)

    model_path = '{0}/models/final_model_amazon.pickle'.format(bucket)

    with open('final_model_amazon.pickle', 'rb') as rfile:
        model = pickle.load(rfile)
    #model = read_model(model_path, fs)
    preds_probas = model.predict(df,  prediction_type='Probability')

    print(preds_probas)
    result = {'probability_0': preds_probas[0][0], 'probability_1': preds_probas[0][1]}

    #return jsonify(result)


from time import time
start = time()
with open('test.json', 'rb') as f:
    data = json.load(f)


online_main(data)
print(time() - start)