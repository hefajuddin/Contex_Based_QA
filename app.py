from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the pre-trained QA model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

@app.route("/")
def home():
    return render_template("index.html")

def load_dynamic_context():
    # Load context from a text file or database
    with open("data/context.txt", "r") as file:
        return file.read()

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    context = load_dynamic_context()
    answer = qa_pipeline(question=user_input, context=context)
    return jsonify({"response": answer['answer']})

if __name__ == "__main__":
    app.run(debug=True, port=5000)