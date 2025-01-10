# Diabetic Patient Readmission Prediction

## Project Overview
This project focuses on predicting hospital readmission for diabetic patients, which is critical for improving healthcare management and resource allocation. The dataset consists of over 100,000 patient records collected from 130 US hospitals between 1999 and 2008. The goal is to identify patients at high risk of readmission, enabling healthcare providers to take preventive measures.

---

## Objectives
- Predict whether a patient will be readmitted to the hospital within 30 days of discharge.
- Analyze key factors contributing to patient readmissions.
- Develop a structured and clean dataset suitable for machine learning models.

---

## Steps Involved

### 1. **Data Exploration**
- **Dataset Overview**: 
  - The dataset contains 50 features and 101,766 observations. These include patient demographics, admission details, medical diagnoses, medication details, and lab test results.
  - The target variable is `readmitted`, which is categorized as:
    - `NO` - Not readmitted.
    - `<30` - Readmitted within 30 days.
    - `>30` - Readmitted after 30 days.

- **Initial Observations**:
  - Many columns contained missing or invalid values.
  - Several categorical features had high cardinality, such as diagnosis codes (`diag_1`, `diag_2`, `diag_3`).
  - Some features like `number_emergency` and `number_outpatient` had a skewed distribution.

---

### 2. **Data Cleaning**
- **Handling Missing Values**:
  - Replaced invalid entries like `"?"` with `NaN`.
  - Dropped columns with >50% missing values, such as `weight`, `payer_code`, and `medical_specialty`.
  - Imputed missing categorical values with the mode.

- **Feature Reduction**:
  - Dropped columns like `encounter_id` and `patient_nbr`, which were unique identifiers and added no predictive value.

- **Outlier Removal**:
  - Used statistical thresholds (mean ± 2.5*std) to detect and remove outliers in numerical features like `time_in_hospital`, `num_lab_procedures`, and `num_medications`.

---

### 3. **Feature Engineering**
- **Target Variable Transformation**:
  - Merged `<30` and `>30` readmission categories into a single class (`1`), leaving `NO` as `0`.

- **Categorical Data Grouping**:
  - Merged less frequent categories in features like `race` (`Hispanic` and `Asian` grouped into `Other`) to reduce cardinality.
  - Diagnoses (`diag_1`, `diag_2`, `diag_3`) were grouped into broad ICD-9 code categories, such as `Circulatory`, `Digestive`, and `Diabetes`.

- **Binary Encoding**:
  - Converted features like `gender` to binary values (`Female=0`, `Male=1`).
  - Grouped medication usage (`No` = 0, all other statuses like `Steady`, `Up`, `Down` = 1).

- **Feature Removal**:
  - Dropped features with negligible variance or minimal impact on predictions, such as `number_outpatient` and `number_emergency`.

- **One-Hot Encoding**:
  - Applied one-hot encoding to categorical variables like `race` and `admission_source_id`.

---
### 4. Data Visualization

To understand the data better, distributions and relationships between features and the target variable (`readmitted`) were visualized using various plots:

- **Bar Plots**: For categorical features like `race`, `gender`, and `readmitted` status.
- **Boxplots and Density Plots**: For numerical features like `time_in_hospital`, `num_lab_procedures`, and `num_medications`.
- **Correlation Heatmaps**: To identify dependencies and relationships between features.

#### Insights:
- Elderly patients (age ≥ 60) formed a significant proportion of the dataset.
- Readmission rates were consistent across genders and racial groups.
- Some features, such as `time_in_hospital`, showed potential predictive importance, while others like `number_emergency` were not impactful.

All the visualizations created during the exploratory data analysis are available in the following folder:

**(DiabeticPatientReadmissionCharts)**

Feel free to explore these visualizations to gain deeper insights into the dataset.


---

### 5. **Model Training**
After preprocessing, the dataset was reduced to 85,040 rows and 35 features. The cleaned and engineered dataset is ready for training machine learning models. Potential models for this project include:
- Logistic Regression for interpretability.
- Random Forest and Gradient Boosting for performance.
- Neural Networks for capturing complex patterns.

Model performance metrics such as accuracy, precision, recall, and AUC will be used to evaluate the results.

---

## Technologies and Tools Used
- **Programming Language**: Python
- **Libraries**:
  - **Data Analysis**: Pandas, NumPy
  - **Visualization**: Matplotlib, Seaborn
  - **Machine Learning**: Scikit-learn
- **Environment**: Jupyter Notebook

---

## Dataset Information
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
- **Features**: 50 (after preprocessing, reduced to 35)
- **Records**: 101,766 (after preprocessing, reduced to 85,040)

---

## Results and Impact
- The cleaned dataset and EDA provided actionable insights into patient readmissions.
- Feature engineering steps improved data quality and suitability for modeling.
- The project lays a foundation for developing predictive systems that can assist healthcare providers in identifying high-risk patients.

---

## Future Work
1. **Model Optimization**:
   - Fine-tune hyperparameters for better performance.
2. **Integration**:
   - Deploy the model as a web service for real-time predictions.
3. **Extended Features**:
   - Incorporate external datasets like socioeconomic factors or lifestyle habits.

---

## Repository
The complete codebase and preprocessing pipeline are available on GitHub. Access it [here](https://github.com/santhosh3760/DiabeticPatientReadmissionPrediction).
