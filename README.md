# Glaucoma Detection System 👁️

A Deep Learning-based web application that detects **Glaucoma** from retinal 
fundus images using a Convolutional Neural Network (CNN). Built with 
TensorFlow/Keras for the model and Flask for the web interface, 
with MongoDB for storing prediction records.

---

## 🌟 Features

- 🔍 **Automated Glaucoma Detection** — Upload a retinal image and get 
  instant AI-powered prediction
- 🧠 **Pre-trained CNN Model** — High-accuracy `.h5` model built with 
  TensorFlow/Keras
- 📊 **Research Visualizations** — Includes:
  - Class distribution plot
  - Training grid (accuracy/loss curves)
  - Confusion matrix
  - ROC curve
  - Precision-Recall curve
- 🗄️ **MongoDB Integration** — All predictions are saved to a database
- 🌐 **Flask Web Interface** — Simple, clean UI for uploading images 
  and viewing results

---

## 🛠️ Tech Stack

| Category         | Technology              |
|------------------|-------------------------|
| Language         | Python 3.x              |
| Deep Learning    | TensorFlow, Keras       |
| Web Framework    | Flask                   |
| Image Processing | OpenCV, PIL (Pillow)    |
| Database         | MongoDB (PyMongo)       |
| Data Science     | Pandas, NumPy           |
| Visualization    | Matplotlib, Seaborn     |
| Frontend         | HTML, CSS, JavaScript   |

---

## 📁 Project Structure

```
GLAUCOMA_PROJECT/
│
├── .vscode/
├── dataset/
├── env/
├── static/
├── templates/
│   └── index.html
│
├── .gitattributes
├── .gitignore
├── app.py
├── glaucoma_model.h5
├── README.md
├── requirements.txt
├── train_model.py
│
├── research_1_distribution.png
├── research_2_training_grid.png
├── research_3_confusion_matrix.png
├── research_4_roc_curve.png
└── research_5_pr_curve.png
```

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8 or higher
- MongoDB installed and running locally
- Git LFS (for downloading the `.h5` model)

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/pawanchaudhariw2006-ctrl/Glaucoma_Project.git
cd Glaucoma_Project
```

**2. Install Git LFS and pull the model**
```bash
git lfs install
git lfs pull
```

**3. Create and activate a virtual environment**
```bash
python -m venv env

# Windows
env\Scripts\activate

# Mac/Linux
source env/bin/activate
```

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

**5. Make sure MongoDB is running**
```bash
# Windows (if MongoDB is installed as a service)
net start MongoDB
```

**6. Run the application**
```bash
python app.py
```

**7. Open in browser**
http://127.0.0.1:5000

---

## 🧠 Model Details

- **Architecture:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow / Keras
- **Input:** Retinal fundus images
- **Output:** Binary classification — `Glaucoma` or `Normal`
- **Model file:** `glaucoma_model.h5` (tracked via Git LFS)

> ⚠️ Note: The compiled metrics will be built upon first prediction. 
> You may see a warning on startup — this is expected behavior.

---

## 📊 Research Visualizations

| File | Description |
|------|-------------|
| `research_1_distribution.png` | Dataset class distribution |
| `research_2_training_grid.png` | Training & validation accuracy/loss |
| `research_3_confusion_matrix.png` | Confusion matrix on test data |
| `research_4_roc_curve.png` | ROC curve with AUC score |
| `research_5_pr_curve.png` | Precision-Recall curve |

---

## 🚀 How It Works

1. User uploads a **retinal fundus image** via the web interface
2. Image is **preprocessed** (resized, normalized) using OpenCV/PIL
3. The **CNN model** runs inference and returns a prediction
4. Result (`Glaucoma` / `Normal`) with confidence score is displayed
5. Prediction record is **saved to MongoDB** for future reference

---

## ⚠️ Disclaimer

This tool is intended for **research and educational purposes only**.  
It is **not a substitute** for professional medical diagnosis.  
Always consult a qualified ophthalmologist for medical advice.

---

## 👨‍💻 Author

**Pawan Chaudhari**  
GitHub: [@pawanchaudhariw2006-ctrl](https://github.com/pawanchaudhariw2006-ctrl)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).