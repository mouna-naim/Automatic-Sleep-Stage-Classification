# Automatic Sleep Stage Classification

## Introduction
This project was developed as part of the Time Series course taught by Laurent Oudre at ENS Paris-Saclay.

Sleep stage classification is a crucial component of sleep research and medical diagnostics, as it helps in understanding sleep patterns, diagnosing sleep disorders, and evaluating overall health. Traditional sleep staging relies on manual annotation by experts, which is time-consuming and prone to subjective variability. To address these limitations, automated sleep stage classification using EEG signals has gained significant interest.

In this project, we implemented and extended the methodology presented in the paper ["Automatic Sleep Stages Classification Using EEG Entropy Features and Unsupervised Pattern Analysis Techniques"](https://www.mdpi.com/1099-4300/16/12/6573). The original work leverages entropy-based metrics extracted from EEG signals combined with unsupervised clustering for classification. We aimed to enhance the performance of this approach by introducing additional statistical features in conjunction with those proposed in the paper. These extra features help refine the classification process by capturing additional signal characteristics and improving accuracy.

## Features & Methodology
1. **EEG Signal Preprocessing**
   - Data sourced from the **SC Sleep-EDF Database [Expanded]**.
   - Fpz-Cz and Pz-Oz EEG channels used.
   - EEG signals segmented into **30-second epochs**.
   
2. **Feature Extraction**
   - Various entropy-based features computed:
     - **Shannon Entropy**
     - **Approximate Entropy (ApEn)**
     - **Sample Entropy (SampEn)**
     - **Multiscale Entropy (MSE)**
     - **Fractal Dimension (FD)**
     - **Detrended Fluctuation Analysis (DFA)**
   - Additional statistical features:
     - Skewness, Kurtosis, Zero-Crossing Rate, and Hjorth Parameters.
   
3. **Feature Selection**
   - **Q-𝛼 Algorithm** used for dimensionality reduction.
   - Extracts **most relevant features** while maintaining classification performance.

4. **Clustering Algorithm**
   - **J-means clustering** for automatic classification of sleep stages:
     - Wakefulness (W)
     - Drowsiness (N1)
     - Light Sleep (N2)
     - Deep Sleep (N3)
     - REM Sleep
   - J-means improves over K-means by handling outliers more effectively.

## Data
- EEG signals sourced from **SC Sleep-EDF Database [Expanded]**.
- Sampled at **100 Hz**.
- Sleep stages merged according to **AASM guidelines**.

## Results & Performance
- Evaluated on **13 subjects**.
- Feature selection with Q-𝛼 algorithm led to:
  - **Significant dimensionality reduction** (from 34 to ~9 features).
  - Improved recall for Deep Sleep (N3) classification.
  - Challenges in detecting N1 stage, consistent with prior literature.
  
## Dependencies
- Python (>=3.7)
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib

## References
- [SC Sleep-EDF Database](https://physionet.org/content/sleep-edfx/1.0.0/sleep-cassette/)
- [Automatic Sleep Stages Classification Using EEG Entropy Features and Unsupervised Pattern Analysis Techniques](https://www.mdpi.com/1099-4300/16/12/6573)


