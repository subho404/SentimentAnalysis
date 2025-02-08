from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from io import BytesIO
import base64

# Load the trained model and TF-IDF vectorizer
clf = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

app = Flask(__name__)

# Store user comments
comments = []

@app.route("/", methods=["GET", "POST"])
def index():
    global comments
    
    if request.method == "POST":
        comment = request.form["comment"]
        if comment: 
            comments.append({"text": comment, "sentiment": None})  # 
    
    return render_template("index.html", comments=comments)


@app.route("/analyze", methods=["POST"])
def analyze():
    global comments
    positive_count = 0
    negative_count = 0

    for comment in comments:
        # Convert comment to TF-IDF format and predict sentiment
        comment_vector = tfidf.transform([comment["text"]])
        prediction = clf.predict(comment_vector)[0]

        # Assign sentiment label
        sentiment = "Positive" if prediction == 1 else "Negative"
        comment["sentiment"] = sentiment

        # Count positive/negative
        if prediction == 1:
            positive_count += 1
        else:
            negative_count += 1

    # Generate a pie chart
    img = BytesIO()
    plt.figure(figsize=(6, 6))
    labels = ["Positive", "Negative"]
    sizes = [positive_count, negative_count]
    colors = ["blue", "red"]
    
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90, shadow=True)
    plt.title("Sentiment Distribution")

    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return jsonify({"comments": comments, "plot_url": plot_url})

if __name__ == "__main__":
    app.run(debug=True)
