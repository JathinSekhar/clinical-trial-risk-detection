# Clinical Trial Risk Detection

AI-powered system to identify potential risks in clinical trials using machine learning and data analysis.

---

## Overview

Clinical trials are complex and expensive processes. Many trials fail due to risks such as patient dropout, safety issues, protocol violations, or operational delays.

This project uses **machine learning techniques to analyze clinical trial data and detect potential risks early**, helping researchers take preventive actions.

---

## Key Features

* AI-based clinical trial risk prediction
* Machine learning model training pipeline
* Dataset generation and preprocessing
* Risk analysis dashboard
* Web interface for interacting with the model

---

## Tech Stack

* Python
* Machine Learning
* Pandas
* Scikit-learn
* Flask
* HTML / CSS
* Git & GitHub

---

## Project Structure

```
clinical-trial-risk-detection
│
├── dashboard.py              # Risk analysis dashboard
├── main.py                   # Main application entry point
├── train_model.py            # ML model training script
├── generate_dataset.py       # Synthetic dataset generation
├── clinical_risk_dataset.csv # Clinical trial dataset
│
├── templates/
│   └── index.html            # Web interface template
│
└── .gitignore
```

---

## Installation

Clone the repository:

```
git clone https://github.com/JathinSekhar/clinical-trial-risk-detection.git
cd clinical-trial-risk-detection
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Project

Train the model:

```
python train_model.py
```

Run the application:

```
python main.py
```

Open in browser:

```
http://localhost:5000
```

---

## Future Improvements

* Integration with real clinical trial datasets
* Advanced deep learning models
* Risk visualization dashboard
* Automated monitoring system

---

## Author

**Jathin Sekhar**

Computer Science Undergraduate
Interested in AI, Backend Development, and Intelligent Systems.
