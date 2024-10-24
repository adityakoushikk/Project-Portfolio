from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

from tensorflow import keras
dlmodel = keras.models.load_model('/Users/kiranbhaskar/Desktop/App/dlmodel2.h5')

with open('dlscaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

import os
import subprocess
import pandas as pd

def process_peptides(peptides):
    with open("input.txt", "w") as f:
        f.write(peptides)

    subprocess.run(["python", "/Users/kiranbhaskar/Desktop/iFeature/iFeature.py", "--file", "/Users/kiranbhaskar/Desktop/App/input.txt", "--type", "CTDC", "--out", "/Users/kiranbhaskar/Desktop/App/CTDC.csv"])
    subprocess.run(["python", "/Users/kiranbhaskar/Desktop/iFeature/iFeature.py", "--file", "/Users/kiranbhaskar/Desktop/App/input.txt", "--type", "CKSAAGP", "--out", "/Users/kiranbhaskar/Desktop/App/CKSAAGP.csv"])
    subprocess.run(["python", "/Users/kiranbhaskar/Desktop/iFeature/iFeature.py", "--file", "/Users/kiranbhaskar/Desktop/App/input.txt", "--type", "CTDD", "--out", "/Users/kiranbhaskar/Desktop/App/CTDD.csv"])
    
    dfgk = pd.read_csv("/Users/kiranbhaskar/Desktop/App/CTDD.csv", sep='\t')
    df2 =  pd.read_csv('/Users/kiranbhaskar/Desktop/App/CKSAAGP.csv', sep='\t')
    df3 =  pd.read_csv('/Users/kiranbhaskar/Desktop/App/CTDC.csv', sep='\t')
    dfgk = dfgk.drop(columns = '#')
    df2 = df2.drop(columns = '#')
    df3 = df3.drop(columns = '#')
    df2 = df2.join(dfgk)
    df3 = df3.join(df2)
    X = df3.values
    finaldf = scaler.transform(X)


    

    return finaldf

def extract_peptide_names(peptides):
    peptide_names = []
    for line in peptides.splitlines():
        if line.startswith('>'):
            peptide_names.append(line[1:])
    return peptide_names

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        peptides = request.form['peptides']
        if len(peptides) > 7:
            input_data = process_peptides(peptides)
            predictions = dlmodel.predict(input_data)
            peptide_names = extract_peptide_names(peptides)
            peptide_predictions = dict(zip(peptide_names, predictions))
            return render_template('results.html', predictions=peptide_predictions)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)

