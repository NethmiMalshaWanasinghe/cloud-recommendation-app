# 🌐 Cloud Recommendation System

This project is a **Multi-Cloud Service Recommendation App** that helps users find the most suitable cloud services based on their requirements. It uses unsupervised machine learning techniques like **DBSCAN** and **LDA** to cluster and analyze cloud service data.

---

## 📌 Features

- 🔍 User input filtering (CPU, memory, bandwidth, provider, service type)
- 📊 Service clustering using **DBSCAN**
- 🧠 Latent topic modeling with **LDA**
- 📈 Interactive Plotly visualizations
- ✅ Filtered, relevant recommendations
- 🧪 Optimization features included
- 🧩 Clean Flask backend and HTML frontend

---

## 🧠 Machine Learning Algorithms Used

- **DBSCAN (Density-Based Spatial Clustering)**  
  → For grouping similar services based on numerical features like CPU, memory, and bandwidth.

- **LDA (Latent Dirichlet Allocation)**  
  → For extracting hidden service topics based on provider/type metadata.

---

## 📂 Files Included

| File Name                        | Description                                  |
|----------------------------------|----------------------------------------------|
| `app.py`                         | Flask backend app                            |
| `recommendation.py`              | Core logic for recommendation                |
| `dbscan_model.pkl`               | Pre-trained DBSCAN model                     |
| `lda_model.pkl`                  | Pre-trained LDA model                        |
| `index.html`                     | Frontend UI                                  |
| `multi_cloud_service_composition.csv` | Dataset used for clustering          |
| `Cloud_Recommendation_System.pdf`| Project report / documentation               |

---

## 🚀 How to Run This Project

### 🔧 Requirements
Install these packages (if not already installed):

```bash
pip install flask pandas numpy scikit-learn plotly

▶️ Run the App
python app.py
Then open your browser and go to:
http://127.0.0.1:5000


📊 Sample Output
Cluster plots of cloud services
Filtered service table
Topic extraction visualization (optional)

📘 Project Purpose
This system is part of a Msc project aimed at improving cloud service selection using intelligent clustering and filtering techniques.

👩‍💻 Author
Nethmi Malsha Wanasinghe
