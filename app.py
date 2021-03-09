from flask import Flask as fl, request, jsonify, session, redirect
import pickle

app = fl(__name__)
app.secret_key="AnExtremelySecretKey123"

model = pickle.load( open( "Models/model.sav", "rb" ) )
vect = pickle.load( open( "Models/vect.sav", "rb" ) )

@app.route('/test')
def hello_world():
    text = request.args['text']
    vect_text = vect.transform([text])
    prediction = {"text": text,
            "bullying": model.predict(vect_text)[0],
            "confidence_bullying": model.predict_proba(vect_text)[0,0] * 100,
            "confidence_not": model.predict_proba(vect_text)[0,1] * 100
            }
    return prediction

if __name__ == "__main__":
    app.run(host="0.0.0.0")