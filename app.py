from flask import Flask,render_template,request 
#.................back-end_code...................#
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
#-------------------------------------------
from os import *
from sys import * 
from os.path import *
from io import BytesIO, IOBase
#-------------------------------------------
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    return 1 - cosine_distance(vector1, vector2)
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix
def generate_summary(rawdocs):
    nltk.download("stopwords")
    stop_words = stopwords.words('english')
    summarize_text = []
    # Step 1 - Read text anc split it
    article = rawdocs.split(". ")
    sentences = []
    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    # sentences.pop() 
    
    raw_sent=len(sentences)
    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    # print("Indexes of top ranked_sentence order are ", ranked_sentence)    
    for i in range(int(.4*len(ranked_sentence))):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
    # Step 5 - Offcourse, output the summarize text
    w=0 
    if len(summarize_text):
        w=len(summarize_text)
    summ_text=". ".join(summarize_text)
    if len(ranked_sentence)>0:
        summ_text+='.'
    return [rawdocs,summ_text,len(rawdocs.split(' ')),len(summ_text.split(' ')),raw_sent,int(.4*len(ranked_sentence))]
    
#.................back-end_code...................#
app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/summary',methods=['GET','POST'])
def summary():
    if request.method=='POST':
        rawtext=request.form['rawtext']
        v=generate_summary(rawtext)
        x,y,z,w,a,b=v[0],v[1],v[2],v[3],v[4],v[5]
    return render_template('summary.html',x=x,y=y,z=z,w=w,a=a,b=b)
if __name__=="__main__":
    app.run(debug=True)