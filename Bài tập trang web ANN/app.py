from flask import Flask, render_template, request
import torch
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== LOAD MODEL =====
model = torch.jit.load(os.path.join(BASE_DIR, "ann_model.pt"))
model.eval()

# ===== ROUTE =====
@app.route("/", methods=["GET","POST"])
def index():
    prediction = None

    if request.method == "POST":
        x_val = request.form.get("x")
        y_val = request.form.get("y")

        try:
            x_val = float(x_val)
            y_val = float(y_val)

            input_tensor = torch.tensor([[x_val, y_val]], dtype=torch.float32)

            with torch.no_grad():
                output = model(input_tensor)
                prob = output.item()

                label = 1 if prob > 0.5 else 0

                prediction = f"Lớp {label} (xác suất: {round(prob*100,2)}%)"

        except:
            prediction = "⚠️ Nhập số hợp lệ!"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)