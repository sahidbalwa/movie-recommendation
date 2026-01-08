# 🎬 Machine Learning–Powered Movie Recommendation System

A **production-grade Machine Learning Movie Recommendation System** built using **Content-Based Filtering (TF-IDF + Cosine Similarity)** on a dataset of **45,000+ movies**, integrated with **real-time TMDB API data** and deployed as a **scalable ML microservice** using **FastAPI** and **Streamlit**.

This project demonstrates **end-to-end ML system design** — from **feature engineering and vectorization** to **API deployment and UI consumption**.

---
**Movie Recommendation System Demo**

![Movie Recommendation System Demo](/Untitled%20design.gif)

---

## 🧠 Machine Learning Problem Statement

> **Given a movie selected by the user, recommend the most similar movies based on textual similarity of metadata.**

This is a **content-based recommendation problem**, solved using **Natural Language Processing (NLP)** and **vector similarity search**.

---

## 🧪 Dataset

* **Size:** 45,000+ movies
* **Source:** Curated movie metadata + TMDB
* **Features used:**

  * Movie title
  * Overview / description
  * Genres
  * Keywords
* **Preprocessing:**

  * Text normalization
  * Tokenization
  * Stopword removal
  * TF-IDF vectorization

---

## 🔬 Machine Learning Approach

### 1️⃣ Feature Engineering

* Combined textual features into a single document per movie
* Applied **TF-IDF Vectorization** to convert text into numerical vectors
* High-dimensional sparse matrix representation

### 2️⃣ Similarity Computation

* Used **Cosine Similarity** to measure angular distance between movie vectors
* Efficient similarity computation using **SciPy sparse matrices**

### 3️⃣ Recommendation Strategy

* Given a movie:

  * Retrieve its TF-IDF vector
  * Compute similarity against all movies
  * Rank movies by similarity score
  * Return top-N recommendations

---

## 📐 Mathematical Intuition

**TF-IDF Weighting**

```
TF-IDF(t, d) = TF(t, d) × log(N / DF(t))
```

**Cosine Similarity**

```
cos(θ) = (A · B) / (||A|| × ||B||)
```

* Measures similarity independent of document length
* Ideal for sparse, high-dimensional NLP vectors

---

## ⚙️ Hybrid Recommendation Architecture

```text
User Query
   ↓
TMDB Movie Matching
   ↓
Movie Metadata
   ↓
TF-IDF Vector Retrieval
   ↓
Cosine Similarity Scoring
   ↓
Top-N ML Recommendations
   +
Genre-Based Discovery (TMDB)
   ↓
Final Ranked Results
```

---

## 🚀 Key ML Features

✅ Content-based filtering (no cold start issue)
✅ Works without user history
✅ Scales to 45k+ movies
✅ Deterministic & explainable recommendations
✅ Real-time enrichment using TMDB API
✅ Fallback-safe inference pipeline

---

## 🛠️ Tech Stack

### Machine Learning

* Scikit-learn
* NumPy
* Pandas
* SciPy (Sparse Matrices)

### Backend (ML Inference API)

* FastAPI
* Uvicorn
* HTTPX
* Python-Dotenv

### Frontend (ML Consumer UI)

* Streamlit

### External Data Source

* TMDB API

---

## 📦 Dependencies

```txt
fastapi==0.111.0
uvicorn==0.30.1
python-dotenv==1.0.1
httpx==0.27.0
pandas==2.2.2
numpy==2.0.1
scipy==1.13.1
scikit-learn==1.5.1
streamlit==1.36.0
```

---



---

## 🔐 Environment Variables

```env
TMDB_API_KEY=your_tmdb_api_key
```





## ▶️ Run Locally

### Backend (ML API)

```bash
uvicorn main:app --reload
```

### Frontend (ML App)

```bash
streamlit run app.py
```

---

🌍 Deployment

Backend (FastAPI ML Service – Render):
🔗 https://movie-recommendation-s49e.onrender.com

Frontend (Streamlit App – Render):
🔗 https://movie-recommendation-kpwflwdaprxersqgappgaj9.streamlit.app/

Cold-start safe

Stateless inference API
---

## 📊 ML Engineering Highlights (Recruiter Section)

✔ End-to-end ML pipeline
✔ NLP feature extraction
✔ Vector similarity search
✔ Sparse matrix optimization
✔ API-based model inference
✔ Production deployment
✔ Clean separation of training & inference

---


## 👨‍💻 Author

**Sahid Balwa**
Machine Learning Engineer | Full-Stack Developer

⭐ If this project helped you, consider starring the repository!
