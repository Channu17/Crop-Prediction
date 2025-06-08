 # 🌾 Farm Mate - Intelligent Agricultural Assistant

A comprehensive machine learning-powered agricultural platform that provides intelligent recommendations for crop selection, fertilizer usage, and soil classification to optimize farming practices.

## 🌐 Live Deployment

- **Web Application**: [https://crop-prediction-cmgu.onrender.com](https://crop-prediction-cmgu.onrender.com)
- **API Endpoint**: [https://crop-prediction-1-6nny.onrender.com](https://crop-prediction-1-6nny.onrender.com)

## 📋 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Machine Learning Models](#machine-learning-models)
- [Datasets](#datasets)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### 🌱 Crop Prediction
- Predicts the most suitable crop based on environmental and soil conditions
- Uses Random Forest Classifier with 99% accuracy
- Considers factors: Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH, and rainfall

### 🧪 Fertilizer Recommendation
- Recommends optimal fertilizer type based on soil and crop conditions
- Supports multiple fertilizer types: Urea, DAP, 28-28, 14-35-14, 20-20, 17-17-17, 10-26-26
- Factors in soil type, crop type, current nutrient levels, and environmental conditions

### 🏞️ Soil Classification
- Classifies soil type from uploaded images using deep learning
- Supports classification of: Alluvial, Black, Clay, Laterite, and Red soil types
- Uses TensorFlow/Keras CNN model for image analysis

## 🛠️ Tech Stack

### Backend
- **FastAPI** - High-performance API framework
- **Python 3.11** - Core programming language
- **TensorFlow/Keras** - Deep learning for soil classification
- **scikit-learn** - Machine learning algorithms
- **OpenCV** - Image processing
- **Pandas & NumPy** - Data manipulation and analysis

### Frontend
- **Streamlit** - Interactive web application
- **HTML/CSS/JavaScript** - Custom web interface
- **Bootstrap** - Responsive UI components

### Deployment
- **Render** - Cloud hosting platform
- **Docker** - Containerization (if applicable)

## 📁 Project Structure

```
f:\Crop Prediction\
├── src/
│   ├── api.py                 # FastAPI backend server
│   ├── app.py                 # Streamlit web application
│   ├── models/                # Trained ML models
│   │   ├── random_forest.pkl           # Crop prediction model
│   │   ├── random_forest_fc.pkl        # Fertilizer recommendation model
│   │   ├── soilClassification.keras    # Soil classification CNN model
│   │   ├── label_encoder.pkl           # Label encoders
│   │   ├── scalar.pkl                  # Feature scalers
│   │   └── ohe.pkl                     # One-hot encoders
│   └── notebooks/             # Jupyter notebooks for model development
│       ├── crop_pred.ipynb
│       ├── fertilizer_pred.ipynb
│       └── soil_classification.ipynb
├── data/
│   ├── dataset/
│   │   ├── Crop_recommendation.csv     # Crop prediction dataset
│   │   └── Fertilizer Prediction.csv   # Fertilizer recommendation dataset
│   └── Soil types/            # Soil type image datasets
├── css/
│   └── style.css              # Custom styling
├── js/
│   └── scripts.js             # JavaScript functionality
├── images/
│   └── farm-welcome-image.jpg # UI assets
├── index.html                 # Main landing page
├── crop_prediction.html       # Crop prediction interface
├── fertilizer_recommendation.html # Fertilizer recommendation interface
├── soil_classification.html   # Soil classification interface
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.11+
- pip package manager
- Virtual environment (recommended)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Crop Prediction"
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv crop-env
   
   # On Windows
   crop-env\Scripts\activate
   
   # On macOS/Linux
   source crop-env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application**
   ```bash
   streamlit run src/app.py
   ```

5. **Run the FastAPI server (separate terminal)**
   ```bash
   uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
   ```

## 📖 Usage

### Web Interface

1. **Home Page**: Navigate to the main interface at `http://localhost:8501`
2. **Crop Prediction**: 
   - Input soil nutrients (N, P, K)
   - Add environmental conditions (temperature, humidity, pH, rainfall)
   - Get crop recommendations
3. **Fertilizer Recommendation**:
   - Select soil type and crop type
   - Input current nutrient levels and environmental conditions
   - Receive fertilizer suggestions
4. **Soil Classification**:
   - Upload soil image (JPG, PNG, JPEG)
   - Get soil type classification with confidence scores

### API Usage

The FastAPI server provides RESTful endpoints for programmatic access:

- **Crop Prediction**: `POST /prediction`
- **Fertilizer Recommendation**: `POST /fertilizerReccommendation`
- **Soil Classification**: `POST /soil_classification`

## 🔌 API Documentation

### Crop Prediction Endpoint

```http
POST /prediction
Content-Type: application/json

{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.8,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.9
}
```

### Fertilizer Recommendation Endpoint

```http
POST /fertilizerReccommendation
Content-Type: application/json

{
  "Temparature": 26,
  "Humidity": 52,
  "Moisture": 38,
  "Soil_Type": "Loamy",
  "Crop_Type": "Maize",
  "Nitrogen": 37,
  "Potassium": 0,
  "Phosphorous": 0
}
```

### Soil Classification Endpoint

```http
POST /soil_classification
Content-Type: multipart/form-data

file: [soil_image.jpg]
```

## 🤖 Machine Learning Models

### 1. Crop Prediction Model
- **Algorithm**: Random Forest Classifier
- **Features**: 7 environmental and soil parameters
- **Accuracy**: 99.09%
- **Classes**: 22 different crops including rice, maize, wheat, etc.
- **Training Data**: 2,200 samples

### 2. Fertilizer Recommendation Model
- **Algorithm**: Random Forest Classifier
- **Features**: Environmental conditions, soil type, crop type, nutrient levels
- **Accuracy**: 100% (training), 100% (testing)
- **Classes**: 7 fertilizer types
- **Training Data**: 99 samples with data augmentation

### 3. Soil Classification Model
- **Algorithm**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras
- **Input**: RGB soil images
- **Classes**: 5 soil types (Alluvial, Black, Clay, Laterite, Red)
- **Architecture**: Deep CNN with multiple convolutional and pooling layers

## 📊 Datasets

### Crop Recommendation Dataset
- **Size**: 2,200 records
- **Features**: N, P, K, temperature, humidity, pH, rainfall
- **Target**: 22 crop types
- **Source**: Agricultural research data

### Fertilizer Prediction Dataset
- **Size**: 99 records
- **Features**: Temperature, humidity, moisture, soil type, crop type, NPK levels
- **Target**: 7 fertilizer types
- **Preprocessing**: One-hot encoding for categorical variables

### Soil Classification Dataset
- **Type**: Image dataset
- **Categories**: 5 soil types
- **Format**: JPG/PNG images
- **Preprocessing**: Image normalization and augmentation

## 🎯 Model Performance

| Model | Algorithm | Accuracy | Features |
|-------|-----------|----------|----------|
| Crop Prediction | Random Forest | 99.09% | 7 numerical features |
| Fertilizer Recommendation | Random Forest | 100% | 22 features (encoded) |
| Soil Classification | CNN | High accuracy | Image processing |

## 🔧 Development

### Model Training
The Jupyter notebooks in `src/notebooks/` contain the complete model development process:

1. **Data preprocessing and exploration**
2. **Feature engineering and selection**
3. **Model training and hyperparameter tuning**
4. **Performance evaluation and validation**
5. **Model serialization and deployment**

### Adding New Features
1. Update the respective notebook with new features
2. Retrain the model with updated data
3. Update the API endpoints in `api.py`
4. Modify the Streamlit interface in `app.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - Initial work and development

## 🙏 Acknowledgments

- Agricultural research community for datasets
- Open source ML libraries and frameworks
- Render platform for hosting

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please reach out:

- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile]
- **GitHub**: [Your GitHub Profile]

---

### 🌟 Star this repository if you found it helpful!

*Made with ❤️ for sustainable agriculture and smart farming*
