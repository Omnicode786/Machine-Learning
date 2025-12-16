<!-- Banner / Title -->
<h1 align="center">🧠 100 Days of Machine Learning – My Notes</h1>

<p align="center">
  Personal notes + summaries for the first 5 videos of CampusX's 
  <a href="https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH" target="_blank">
    100 Days of Machine Learning
  </a>.
  Written as if I’m explaining concepts to my future self.
</p>

<!-- Badges Row -->
<p align="center">
  <img src="https://img.shields.io/badge/Status-Learning%20In%20Progress-22c55e?style=for-the-badge&logo=bookstack&logoColor=white" />
  <img src="https://img.shields.io/badge/Focus-Machine%20Learning-0ea5e9?style=for-the-badge&logo=dependabot&logoColor=white" />
  <img src="https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Playlist-CampusX-8b5cf6?style=for-the-badge&logo=youtube&logoColor=white" />
</p>

---

## 📚 Table of Contents

- [About This Repo](#-about-this-repo)
- [Tech & Tools](#-tech--tools)
- [Video Notes](#-video-notes)
  - [Video 1 – What is Machine Learning?](#video-1--what-is-machine-learning)
  - [Video 2 – AI vs ML vs DL](#video-2--ai-vs-ml-vs-dl)
  - [Video 3 – Types of Machine Learning](#video-3--types-of-machine-learning)
  - [Video 4 – Batch / Offline Learning](#video-4--batch--offline-learning)
  - [Video 5 – Online / Incremental Learning](#video-5--online--incremental-learning)
- [How I Use These Notes](#-how-i-use-these-notes)
- [Next Steps](#-next-steps)

---

## 📝 About This Repo

This repo is my personal knowledge base for the first 5 videos of the **100 Days of Machine Learning** playlist by CampusX.  
I’m treating this like a learning log plus a quick-ref guide for revising core ML concepts.

---

## 🛠 Tech & Tools

<p align="left">
  <img src="https://img.shields.io/badge/Editor-VS%20Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white" />
  <img src="https://img.shields.io/badge/Notebook-Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/ML-Learning%20Concepts-22c55e?style=for-the-badge&logo=serverless&logoColor=white" />
</p>

> 🌐 Playlist:  
> **100 Days of Machine Learning – CampusX**  
> https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH

---

## 📼 Video Notes

### Video 1 – What is Machine Learning?

**Idea in my own words:**  
Machine Learning = teaching computers to learn patterns from data instead of me hardcoding rules.

**My key bullets:**
- ML builds models from **training data** and uses them to **predict** or **decide** without explicit rules.
- Traditional programming: *rules + data → output*.  
  ML: *data + output → model → predictions*.
- Super useful when rules are too complex or constantly changing.

**Examples that stuck with me:**
- Spam filter, medical diagnosis, speech recognition, computer vision.

---

### Video 2 – AI vs ML vs DL

**Mental model:**  
> **AI ⟶ ML ⟶ DL** (big umbrella → subset → sub-subset)

- **AI:** Any technique that makes machines act “smart”.
- **ML:** Subset of AI where systems **learn from data**.
- **DL:** Subset of ML that uses **neural networks** and **lots of data**.

**When I’d use what:**
- Use *ML* for structured/tabular problems (regression, classification).
- Use *DL* for images, audio, text, or huge complex datasets.
- DL shines once data + compute are big enough.

---

### Video 3 – Types of Machine Learning

**Main categories:**
- **Supervised Learning:** I have input **X** and labeled output **y**.
  - **Regression:** numeric output (e.g., salary prediction).
  - **Classification:** categorical output (e.g., placed vs not placed).
- **Unsupervised Learning:** Only inputs, **no labels**.
  - **Clustering:** group similar items (customer segments, “beer & diapers” story).
  - **Dimensionality Reduction:** compress many features into fewer dimensions.
  - **Anomaly Detection:** catch outliers (fraud, defects).
  - **Association Rules:** find “items bought together”.

- **Semi-supervised Learning:** A small labeled set + a big unlabeled set.
  - Example: Google Photos auto-grouping faces after I label a few.

- **Reinforcement Learning:** Learn by **trial and error** using rewards.
  - Agent interacts with environment → gets reward/punishment → updates policy.
  - Famous example: AlphaGo learning to play Go at superhuman level.

---

### Video 4 – Batch / Offline Learning

**Definition in my words:**  
Train once on a big historical dataset → deploy model → it stays static until I retrain.

**Pipeline:**
1. Collect full dataset.
2. Train model locally / in a training environment.
3. Deploy trained model to production.
4. If data distribution changes → retrain from scratch and redeploy.

**Pros:**
- Simple mental model and deployment.
- Good for problems where patterns don’t change quickly.

**Cons:**
- Needs enough compute/RAM to fit + train on full data.
- Model **doesn’t adapt** to new patterns automatically.
- Long retraining cycles can be expensive.

---

### Video 5 – Online / Incremental Learning

**Definition in my words:**  
Model keeps learning **as new data streams in**, instead of only learning once.

**High-level flow:**
1. Start with initial model.
2. Stream new data in mini-batches or one-by-one.
3. Call incremental updates (e.g., `partial_fit`-style).
4. Model gradually adapts to the latest data.

**Where it shows up:**
- Chatbots and assistants (learning from live conversations).
- Keyboard suggestions (SwiftKey style).
- Recommendation systems (YouTube, e-commerce).
- Streaming fraud detection.

**Key concept: learning rate**
- Too high → model forgets older knowledge too fast.
- Too low → model is slow to adapt.
- Needs a balance between memory and adaptability.

**Challenges:**
- Much harder to monitor in production.
- Risk of **catastrophic forgetting** if only recent data dominates.
- Vulnerable to malicious data poisoning.
- Needs proper logging, monitoring, and rollback strategy.

---

## 🔍 How I Use These Notes

- As a **quick revision doc** when I forget terminology.
- As a **concept map** before diving into coding ML algorithms.
- To decide **which learning setup** (batch vs online, supervised vs unsupervised) fits a new problem.

---

## 🚀 Next Steps

- Start implementing simple supervised models (regression & classification).
- Play with a small **online learning** demo using Python.
- Add code snippets and diagrams to this repo as I progress.
- Extend notes beyond Video 5 as I move forward in the playlist.

---

<p align="center">
  Made with 🧠, ☕, and lots of curiosity.
</p>
