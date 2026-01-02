# Road Surface Roughness Detection

A machine learning-powered Streamlit application for detecting and classifying road surface roughness conditions using smartphone sensor data collected via Sensor Logger (accelerometer, gyroscope, and GPS).

## 🚀 Features

- **Multiple ML Models**: Choose from three trained models:
  - XGBoost (Model 2)
  - Random Forest (Model 1)
  - Support Vector Machine (Model 4)

- **Intelligent Feature Extraction**: Automatically processes sensor data from accelerometer, gyroscope, and GPS CSV files

- **Comprehensive Visualizations**:
  - Interactive map showing road segments with color-coded conditions
  - Pie chart distribution of road conditions
  - Bar chart summary of surface conditions
  - Detailed prediction results with IRI (International Roughness Index) estimates

- **Road Condition Classification**: Predicts three categories:
  - **Smooth** (IRI ≤ 2.0)
  - **Fair** (2.0 < IRI ≤ 3.5)
  - **Rough** (IRI > 3.5)

## 📋 Prerequisites

- Python 3.10
- pip package manager

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd RoadRoughness
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Required Packages

The following packages are required (automatically installed via `requirements.txt`):

- `streamlit` - Web application framework
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `scikit-learn` - Machine learning library
- `matplotlib` - Plotting library
- `seaborn` - Statistical visualization
- `folium` - Map visualization
- `streamlit-folium` - Folium integration for Streamlit
- `xgboost` - Gradient boosting framework
- `joblib` - Model serialization
- `plotly` - Interactive plotting

## 🎯 Usage

1. **Launch the application**:
   ```bash
   streamlit run app.py
   ```

2. **Prepare your data**: 
   - Use **Sensor Logger** (or compatible sensor logging app) to collect data while driving
   - Export the sensor data as CSV files and create a ZIP file containing:
     - Accelerometer data (filename must contain "accelerometer")
     - Gyroscope data (filename must contain "gyroscope")
     - GPS/Location data (filename must contain "location")

3. **Upload and analyze**:
   - Use the file uploader to select your ZIP file
   - Select your preferred ML model from the sidebar
   - Configure visualization options (map, charts, etc.)
   - View predictions and analysis results

## 📊 Data Format

### Data Collection

Sensor data should be collected using **Sensor Logger** (or a compatible sensor logging application) on your smartphone. The app records accelerometer, gyroscope, and GPS/location data simultaneously during vehicle movement.

### Input CSV Requirements

Each CSV file exported from Sensor Logger should have the following structure:

- **First column**: Timestamp (in milliseconds)
- **Subsequent columns**: Sensor readings (x, y, z axes for accelerometer/gyroscope; latitude, longitude, speed, altitude for GPS)

The application automatically:
- Detects and processes timestamp formats
- Merges sensor data using temporal alignment
- Filters segments based on minimum speed (5 km/h)
- Extracts features in 200-sample windows

### Extracted Features

The application extracts the following features for each road segment:

- `mean_accel_y`: Mean acceleration in Y-axis
- `std_accel_y`: Standard deviation of Y-axis acceleration
- `rms_accel_y`: Root mean square of Y-axis acceleration
- `peak2peak_accel_y`: Peak-to-peak amplitude
- `mean_speed`: Average speed (converted to km/h)
- `elevation_change`: Change in altitude
- `gyro_y_std`: Standard deviation of Y-axis gyroscope
- `gyro_x_std`: Standard deviation of X-axis gyroscope
- `latitude`: Average latitude
- `longitude`: Average longitude

## 📁 Project Structure

```
RoadRoughness/
├── app.py                 # Main Streamlit application
├── model_utils.py         # Feature extraction utilities
├── models/                # Trained ML models
│   ├── xgb2_model.pkl     # XGBoost model
│   ├── rf1_model.pkl      # Random Forest model
│   ├── svm4_model.pkl     # SVM model
│   └── label_encoder.pkl  # Label encoder for XGBoost
├── requirements.txt       # Python dependencies
├── runtime.txt           # Python version specification
└── README.md             # This file
```

## 🔬 Technical Details

### Model Information

- **XGBoost Model**: Uses label encoding for categorical outputs
- **Random Forest & SVM Models**: Direct categorical predictions
- All models are pre-trained and ready to use

### IRI Estimation

The application estimates International Roughness Index (IRI) using the formula:
```
IRI = 5.132 × RMS_Accel_Y + 1.112
```

IRI values are then classified into road condition categories for comparison with model predictions.

### Data Processing Pipeline

1. **Data Loading**: Reads accelerometer, gyroscope, and GPS CSV files
2. **Temporal Alignment**: Merges sensor data using nearest-neighbor temporal matching
3. **Feature Extraction**: Processes data in sliding windows (200 samples)
4. **Filtering**: Removes segments with speed < 5 km/h
5. **Prediction**: Applies selected ML model to extracted features
6. **Visualization**: Generates maps, charts, and summary statistics

## 🗺️ Visualization Features

- **Interactive Map**: Color-coded markers showing road condition at each segment
  - Green: Smooth
  - Blue: Fair
  - Red: Rough

- **Pie Chart**: Distribution of predicted road conditions

- **Bar Chart**: Count of segments by condition type

- **Data Table**: Detailed results with predictions, IRI estimates, and true labels

## ⚙️ Configuration

Use the sidebar to customize:
- **Model Selection**: Choose between XGBoost, Random Forest, or SVM
- **Show/Hide Map**: Toggle interactive map visualization
- **Show/Hide Pie Chart**: Toggle pie chart display
- **Show/Hide Bar Chart**: Toggle bar chart display
- **Show IRI & True Label**: Toggle detailed metrics in results table

## 🐛 Troubleshooting

- **"No valid features extracted"**: Ensure your ZIP file contains correctly named CSV files with valid sensor data
- **"File not found"**: Check that CSV filenames contain keywords: "accelerometer", "gyroscope", and "location"
- **Import errors**: Verify all dependencies are installed using `pip install -r requirements.txt`

## 📝 Notes

- The application processes data in 200-sample windows
- Minimum speed threshold: 5 km/h (segments below this are filtered out)
- GPS tolerance: 200ms for temporal alignment
- Accelerometer/Gyroscope tolerance: 50ms for temporal alignment
- Timezone adjustment: +8 hours applied to timestamps

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📄 License

[Add your license information here]

---

**Built with ❤️ using Streamlit and Machine Learning**

