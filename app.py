from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the pre-trained QA model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    # Use a predefined context to answer factual questions
    context = (
        "Beijing is the capital of China. Dhaka is the capital of Bangladesh. "
        "Washington, D.C., is the capital of the United States."
    )
    answer = qa_pipeline(question=user_input, context=context)
    return jsonify({"response": answer['answer']})

if __name__ == "__main__":
    app.run(debug=True)