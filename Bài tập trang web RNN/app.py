from flask import Flask, render_template, request
import torch
import os
import numpy as np

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== LOAD MODEL =====
model = torch.jit.load(os.path.join(BASE_DIR, "rnn_model.pt"))
model.eval()

seq_length = 10
hidden_size = 20

# ===== INIT HIDDEN =====
def init_hidden(batch_size=1):
    return torch.zeros(1, batch_size, hidden_size)

# ===== TEXT → SEQUENCE =====
def parse_input(text):
    try:
        nums = [float(x) for x in text.split()]
        return nums
    except:
        return None

# ===== ROUTE =====
@app.route("/", methods=["GET","POST"])
def index():
    prediction = None

    if request.method == "POST":
        text = request.form.get("sequence")

        nums = parse_input(text)

        if nums and len(nums) == seq_length:
            seq = torch.tensor(nums, dtype=torch.float32).view(1, seq_length, 1)
            hidden = init_hidden(1)

            with torch.no_grad():
                output, hidden = model(seq, hidden)
                prediction = round(output.item(), 4)
        else:
            prediction = "⚠️ Nhập đúng 10 số!"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)