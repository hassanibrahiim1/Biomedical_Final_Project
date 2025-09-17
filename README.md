# ECG Arrhythmia Detection and Classification

## Project Overview
This project is a comprehensive implementation of an ECG signal analysis pipeline for the detection and classification of cardiac arrhythmias. Developed as a Final Project for the Biomedical Engineering program, Faculty of Engineering, Alexandria University, it guides through the complete process from raw signal to diagnosis using the MIT-BIH Arrhythmia Database.

The system performs:
1.  **Dataset Exploration & Visualization:** Loading and understanding the MIT-BIH database.
2.  **Signal Preprocessing:** Removing noise, baseline wander, and powerline interference.
3.  **Feature Extraction:** Detecting R-peaks and calculating heart rate & RR intervals.
4.  **Arrhythmia Classification:** Using machine learning to classify signals as normal or abnormal.

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Dataset](#dataset)
3.  [Installation & Dependencies](#installation--dependencies)
4.  [Project Structure](#project-structure)
5.  [Usage](#usage)
6.  [Methodology](#methodology)
7.  [Results](#results)
8.  [Team](#team)
9.  [License](#license)

## Dataset
This project uses the **MIT-BIH Arrhythmia Database**.
*   **Source:** [PhysioNet - MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
*   **Download Instructions:** Download the full ZIP archive from the link above. Extract it and ensure the records (e.g., `100`, `101`, `102`) and their corresponding files (`.dat`, `.hea`, `.atr`) are accessible by the code.
*   **Description:** The database contains 48 half-hour excerpts of two-channel ambulatory ECG recordings, with annotations indicating the heart rhythm for each beat.

## Installation & Dependencies
This project is implemented in Python 3.8+. It requires the following libraries:

| Library | Purpose |
| :--- | :--- |
| `wfdb` | Reading MIT-BIH ECG data and annotations |
| `numpy` | Numerical computations and array operations |
| `scipy` | Signal processing and filtering |
| `matplotlib` | Data and signal visualization |
| `pandas` | Feature engineering and data manipulation |
| `scikit-learn` | Machine learning models and evaluation metrics |
| `tqdm` (optional) | Progress bars for long loops |

### Installation
1.  Clone this repository:
    ```bash
    git clone <your-repository-url>
    cd ecg-arrhythmia-detection
    ```
2.  Install the required packages using pip:
    ```bash
    pip install wfdb numpy scipy matplotlib pandas scikit-learn tqdm
    ```

## Project Structure
The project is organized into a Jupyter Notebook (.ipynb) that follows the four main sections outlined in the project description.
ecg-arrhythmia-detection/
│
├── ECG_Arrhythmia_Detection.ipynb # Main Jupyter Notebook
├── data/ # Directory for MIT-BIH dataset (not included in repo)
│ ├── 100.data
│ ├── 100.hea
│ ├── 100.atr
│ └── ...
├── utils.py # (Optional) Helper functions
├── requirements.txt # List of dependencies
└── README.md # This file

## Usage
1.  **Download the Dataset:** Follow the instructions in the [Dataset](#dataset) section and place the unzipped files in a `data/` folder within the project directory.
2.  **Open the Notebook:** Launch Jupyter Notebook or Jupyter Lab.
    ```bash
    jupyter notebook
    ```
3.  **Run the Code:** Execute the cells in `ECG_Arrhythmia_Detection.ipynb` sequentially. The notebook is divided into clear sections:
    *   **Section 1:** Loads and visualizes records 100, 101, and 102.
    *   **Section 2:** Applies preprocessing filters (Baseline, Notch, Bandpass) and visualizes the results.
    *   **Section 3:** Detects R-peaks, calculates RR intervals and heart rate, and plots the findings.
    *   **Section 4:** Classifies heartbeats using a machine learning model and evaluates its performance.

## Methodology
### 1. Dataset Exploration
*   Used the `wfdb` library to read signals and annotations.
*   Visualized 10-second segments of leads (e.g., MLII) with annotated beats.
*   Analyzed the distribution of heartbeat types (N, V, L, R, A, etc.) in the selected segments.

### 2. Signal Preprocessing
A multi-stage filtering approach was implemented:
*   **Baseline Wander Removal:** Applied a high-pass filter (cutoff ~0.5 Hz) using FFT or a Butterworth filter.
*   **Powerline Interference Removal:** Designed a notch filter at 50 Hz (or 60 Hz) to suppress powerline noise.
*   **High-Frequency Noise Reduction:** Used a bandpass filter (e.g., 3-45 Hz) to smooth the signal while preserving the QRS complex morphology.

### 3. Feature Extraction (R-Peak Detection)
*   Employed the `scipy.signal.find_peaks()` function on the preprocessed signal.
*   Calculated RR intervals (the time between consecutive R-peaks).
*   Derived features such as Instantaneous Heart Rate and Heart Rate Variability (HRV).

### 4. Arrhythmia Classification
*   **Simple Rule-Based Method:** Classified based on average heart rate (Bradycardia: <60 bpm, Tachycardia: >100 bpm, Normal: 60-100 bpm).
*   **Machine Learning Model:** Extracted features from RR intervals (e.g., mean, standard deviation, NN50, pNN50) and trained a classifier (e.g., Random Forest, Logistic Regression) to distinguish between normal and abnormal beats/rhythms.
*   **Evaluation:** Assessed model performance using metrics like Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.

## Results
The project successfully demonstrates a complete workflow for ECG analysis. Key outcomes include:
*   Clean, noise-free ECG signals after the preprocessing pipeline.
*   Accurate detection of R-peaks validated by visual inspection.
*   Calculation of heart rate and observation of variability from RR intervals.
*   A machine learning model capable of classifying arrhythmias with a reported performance (e.g., >90% accuracy on a test set, depending on implementation).
*   Detailed visualizations and commentary are provided at each step within the Jupyter Notebook.

## Team
This project was completed by:
- [Hassan Ibrahim]
- [Eslam Sameh]
- [Mohamed Saad]


## License
This project is created for academic purposes. The MIT-BIH Arrhythmia Database is available under the [PhysioNet Open Data License](https://physionet.org/content/mitdb/1.0.0/).
