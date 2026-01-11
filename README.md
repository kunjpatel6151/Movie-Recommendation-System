# 🎬 Movie Recommendation System

A content-based movie recommendation system built with machine learning and a Streamlit web interface. The system analyzes movie plots, creators, and numerical features to provide personalized recommendations.

## 📋 Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [ML Pipeline](#ml-pipeline)
- [Data Sources](#data-sources)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Hybrid Recommendation Algorithm**: Combines story similarity (TF-IDF on plot + tagline), creator similarity (TF-IDF on cast, directors, genres, etc.), and numerical scores
- **Interactive Web Interface**: Clean Streamlit app with searchable movie selection
- **Real-time Recommendations**: Get top 10 movie recommendations instantly
- **Poster Integration**: Displays movie posters from TMDB
- **Responsive Design**: Works on wide screens with grid layout

## 🚀 Demo

The application is live and can be accessed by running the Streamlit app locally.

## 🛠 Installation

### Prerequisites

- Python 3.10+
- Git

### Clone the Repository

```bash
git clone https://github.com/kunjpatel6151/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

### Install Dependencies

```bash
pip install streamlit pandas numpy scipy scikit-learn joblib
```

## 📖 Usage

### Running the Web Application

```bash
streamlit run streamlit_app.py
```

1. Open your browser to `http://localhost:8501`
2. Select one or more movies you like from the searchable dropdown
3. Click "Recommend" to get personalized suggestions
4. View the top 10 recommended movies in a grid layout

### Understanding the ML Pipeline

The system uses a pre-trained model with the following components:

- **Story Similarity**: TF-IDF vectorization of movie overviews and taglines
- **Creator Similarity**: TF-IDF on weighted metadata (directors, cast, writers, producers, genres, etc.)
- **Numerical Score**: Scaled combination of ratings, popularity, budget, etc.
- **Final Score**: Weighted combination (55% story + 25% creator + 20% numerical)

## 🔬 ML Pipeline

The machine learning pipeline consists of three Jupyter notebooks:

1. **Numeric Score** (`Numeric Score.ipynb`)

   - Processes numerical features (ratings, votes, budget, etc.)
   - Applies min-max scaling
   - Output: `movies_with_numeric_score.csv`

2. **Content Score** (`Content Score.ipynb`)

   - Story channel: TF-IDF on (overview + tagline)
   - Creator channel: TF-IDF on metadata with director boosting
   - Outputs: TF-IDF matrices and vectorizers

3. **Final Score** (`Final Score.ipynb`)
   - Combines similarities and scores
   - Implements recommendation logic
   - Validates the complete pipeline

## 📊 Data Sources

- **Kaggle Dataset**: [TMDB Movies Daily Updates](https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates) - Raw movie data from TMDB
- **TMDB API**: Movie metadata, posters, ratings
- **Data Cleaning**: `Data Cleaning.ipynb` processes raw TMDB data
- **Preprocessed Data**:
  - `movies_with_content_meta.csv`: Main dataset with all features
  - `movies_with_numeric_score.csv`: Numerical scores
  - TF-IDF artifacts: `.joblib` vectorizers and `.npz` sparse matrices

## 🏗 Project Structure

```
Movie-Recommendation-System/
├── streamlit_app.py              # Main web application
├── movies_with_content_meta.csv  # Main movie dataset
├── movies_with_numeric_score.csv # Numerical scores
├── story_tfidf_vectorizer.joblib # Story TF-IDF vectorizer
├── story_tfidf.npz              # Story TF-IDF matrix
├── creators_tfidf.joblib        # Creator TF-IDF vectorizer
├── creators_tfidf.npz           # Creator TF-IDF matrix
├── Numeric Score.ipynb          # Numerical feature processing
├── Content Score.ipynb          # Content-based feature extraction
├── Final Score.ipynb            # Recommendation algorithm
├── Data Cleaning.ipynb          # Data preprocessing
├── TMDB_all_movies.csv          # Raw TMDB data
└── README.md                    # This file
```

## 🛠 Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: TF-IDF vectorization
- **SciPy**: Sparse matrix operations
- **Joblib**: Model serialization
- **TMDB API**: Movie data and posters

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TMDB for providing comprehensive movie data
- The movie recommendation community for inspiration
- Open-source libraries that made this project possible

---

**Note**: This project uses pre-computed ML artifacts. To retrain models, run the Jupyter notebooks in order: Data Cleaning → Numeric Score → Content Score → Final Score.</content>
<parameter name="filePath">d:\B.Tech CSE\Movie Recommendation System\README.md
