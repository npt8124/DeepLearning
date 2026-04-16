from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import os

from model import CatDogCNN, PlantCNN, CIFAR_CNN

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = "cpu"

# ================= LOAD MODEL =================
catdog_model = CatDogCNN()
catdog_model.load_state_dict(torch.load(os.path.join(BASE_DIR,"models/cat_dog_model.pth"), map_location=device))
catdog_model.eval()

plant_model = PlantCNN(num_classes=15)  # ⚠️ đúng của bạn
plant_model.load_state_dict(torch.load(os.path.join(BASE_DIR,"models/plant_model.pth"), map_location=device))
plant_model.eval()

cifar_model = CIFAR_CNN()
cifar_model.load_state_dict(torch.load(os.path.join(BASE_DIR,"models/cifar10_model.pth"), map_location=device))
cifar_model.eval()

# ================= LABEL =================
catdog_classes = ["Cat","Dog"]

cifar_classes = ["Airplane","Automobile","Bird","Cat","Deer",
                 "Dog","Frog","Horse","Ship","Truck"]

plant_classes = ["Pepper__bell___Bacterial_spot",
                 "Pepper__bell___healthy",
                 "Potato___Early_blight",
                 "Potato___healthy",
                 "Potato___Late_blight",
                 "Tomato__Target_Spot",
                 "Tomato__Tomato_mosaic_virus",
                 "Tomato__Tomato_YellowLeaf__Curl_Virus",
                 "Tomato_Bacterial_spot",
                 "Tomato_Early_blight",
                 "Tomato_healthy",
                 "Tomato_Late_blight",
                 "Tomato_Leaf_Mold",
                 "Tomato_Septoria_leaf_spot",
                 "Tomato_Spider_mites_Two_spotted_spider_mite"
                 ]

# ================= TRANSFORM =================
transform_128 = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_cifar = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),
                         (0.2023,0.1994,0.2010))
])

# ================= ROUTE =================
@app.route("/", methods=["GET","POST"])
def index():
    prediction = None
    confidence = None
    img_path = None
    all_results = None

    if request.method == "POST":

        file = request.files["file"]

        if file:
            filename = file.filename
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)

            img_path = "uploads/" + filename

            img = Image.open(path).convert("RGB")

            with torch.no_grad():

                # ===== CATDOG =====
                img1 = transform_128(img).unsqueeze(0)
                out1 = catdog_model(img1)
                prob1 = torch.softmax(out1, dim=1)
                conf1, pred1 = torch.max(prob1, 1)

                # ===== PLANT =====
                img2 = transform_128(img).unsqueeze(0)
                out2 = plant_model(img2)
                prob2 = torch.softmax(out2, dim=1)
                conf2, pred2 = torch.max(prob2, 1)

                # ===== CIFAR =====
                img3 = transform_cifar(img).unsqueeze(0)
                out3 = cifar_model(img3)
                prob3 = torch.softmax(out3, dim=1)
                conf3, pred3 = torch.max(prob3, 1)

                results = [
                    {"model":"CatDog","label":catdog_classes[pred1.item()],"conf":conf1.item()},
                    {"model":"Plant","label":plant_classes[pred2.item()],"conf":conf2.item()},
                    {"model":"CIFAR","label":cifar_classes[pred3.item()],"conf":conf3.item()},
                ]

                best = max(results, key=lambda x: x["conf"])

                prediction = best["label"]
                confidence = round(best["conf"]*100,2)

                all_results = [
                    {**r, "conf": round(r["conf"]*100,2)} for r in results
                ]

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        img_path=img_path,
        all_results=all_results
    )
    
if __name__ == "__main__":
    app.run(debug=True)