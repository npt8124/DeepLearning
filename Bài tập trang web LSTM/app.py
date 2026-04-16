from flask import Flask, render_template, request
import torch
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== LOAD MODEL (.pt) =====
model = torch.jit.load(os.path.join(BASE_DIR, "lstm_model.pt"))
model.eval()

# ===== VOCAB =====
vocab = {'I': 1, 'like': 2, 'to': 3, 'music': 4,
         'not': 5, 'sad': 6, 'happy': 7}

# ===== TEXT → TENSOR =====
def text_to_tensor(text):
    words = text.split()
    seq = [vocab.get(w, 0) for w in words]
    return torch.tensor(seq, dtype=torch.long).unsqueeze(0)

# ===== ROUTE =====
@app.route("/", methods=["GET","POST"])
def index():
    prediction = None

    if request.method == "POST":
        text = request.form["text"]

        if text:
            seq = text_to_tensor(text)

            with torch.no_grad():
                output = model(seq)
                _, pred = torch.max(output, 1)

                prediction = "Tích cực" if pred.item() == 1 else "Tiêu cực"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)