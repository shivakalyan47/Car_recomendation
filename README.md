# AI-Powered Car Recommender 🚗🤖

An advanced, interactive web application designed to recommend ideal passenger vehicles based on user preferences. This system utilizes a **K-Nearest Neighbors (KNN)** machine learning algorithm, Scikit-learn, Pandas, Plotly, and Streamlit, providing a polished and comprehensive implementation suitable for a **final-year engineering mini-project**.

---

## 🌟 Key Features

1. **AI Recommendation Engine**: Uses Scikit-learn's `NearestNeighbors` to calculate multi-dimensional distances between user queries and vehicle data.
2. **Dual-Matching Architecture**:
   - **Strict Filtering**: Combines exact categorical matching (e.g., matching Petrol Automatic 5-Seater) with relative numerical KNN distance scoring (Budget, Mileage, and Engine size).
   - **Soft Matching**: Fits the mathematical model across all combined features (scaled numeric columns + one-hot encoded categorical columns) to recommend optimal alternatives if exact profiles are unavailable.
3. **Flexible Distance Metrics**: Toggle between **Euclidean Distance** (absolute spatial separation) and **Cosine Similarity** (directional alignment of preference vectors).
4. **Match Confidence Score**: Displays a percentage likeness score (e.g. `96.5% Match`) based on normalized vector metrics.
5. **Interactive Data Dashboard (Plotly)**: Premium, responsive visualizations including:
   - Fuel Type market share donut charts
   - Brand pricing distributions
   - Mileage vs. Range comparison bar graphs
   - Best-budget vehicle analysis
6. **Side-by-Side Car Comparison**: Compare any two cars side-by-side with specification metrics and conditional green/red formatting highlighting the better vehicle.
7. **Browse & Search Catalog**: Interactive database table sorting, filtering, and live search.

---

## 📂 Folder Structure

```
IML aat/
│
├── data/
│   └── cars.csv                    # Cleaned multi-brand car dataset (130+ entries)
│
├── utils/
│   ├── preprocessing.py            # Feature engineering (MinMaxScaler & One-Hot Maps)
│   ├── knn_model.py                # KNN NearestNeighbors model core logic
│   └── visualizations.py           # Premium interactive Plotly chart generators
│
├── app.py                          # Multi-tab Streamlit dashboard frontend
├── generate_data.py                # Script to generate dataset
├── requirements.txt                # Project dependency requirements
└── README.md                       # Documentation and Setup instructions
```

---

## 🧠 Machine Learning & Preprocessing Architecture

### 1. Preprocessing Pipeline (`utils/preprocessing.py`)
- **Numerical Scaling**: Columns like `Price`, `Mileage`, `Seats`, and `Engine` are normalized into $[0, 1]$ using a `MinMaxScaler`. This prevents larger magnitude metrics (like engine cc) from dominating smaller magnitude metrics (like seats or price) during distance calculations.
- **Categorical Encoding**: Fields like `Fuel Type` and `Transmission` are one-hot encoded into binary indicator vectors (`Fuel_Petrol`, `Fuel_Electric`, `Trans_Manual`, etc.).

### 2. Distance Calculations (`utils/knn_model.py`)
- **Euclidean Distance**:
  $$d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^{n} (u_i - v_i)^2}$$
  Perfect for absolute coordinate differences in normalized preference spaces.
- **Cosine Distance / Similarity**:
  $$d_{\cos}(\mathbf{u}, \mathbf{v}) = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$
  Measures the angular difference, focusing on relative alignment and proportion of preferences.
- **Confidence Score**: Normalized inverse distance converted directly into percentage matching values.

---

## 🛠️ Step-by-Step Local Setup Instructions

### Prerequisites
Make sure Python 3.9+ is installed on your computer.

### 1. Clone or Open Project
Ensure all project files are located in your workspace directory:
`c:\Users\shiva\Downloads\IML aat`

### 2. Install Dependencies
Open standard terminal or PowerShell and run the following command to install required modules:
```bash
pip install -r requirements.txt
```

### 3. Generate Dataset
If `data/cars.csv` is not present, run the generation script:
```bash
python generate_data.py
```

### 4. Run the Streamlit Application
Start the Streamlit local dev server by running:
```bash
streamlit run app.py
```

Streamlit will compile and launch the dashboard directly in your default web browser at `http://localhost:8501`.

---

## ☁️ Deployment to Streamlit Cloud

Streamlit provides free cloud hosting optimized for sharing machine learning portfolios. Follow these steps to host your mini project:

1. **Push Code to GitHub**:
   - Create a new public repository on GitHub.
   - Commit all files (including `requirements.txt`, `utils/`, `data/`, `app.py`, `generate_data.py`).
   - Push code to the repository.
   
2. **Deploy on Streamlit Community Cloud**:
   - Visit [Streamlit Share](https://share.streamlit.io/) and log in with your GitHub account.
   - Click the **"New app"** button.
   - Select your repository, branch (`main`/`master`), and main file path (`app.py`).
   - Click **"Deploy"**.

Streamlit Cloud will configure the python container, install all packages from your `requirements.txt`, execute `app.py`, and provide you with a permanent, shareable public URL!
