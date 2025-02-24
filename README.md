# Image Classification with PCA and Machine Learning

## Project Overview
This project involves classifying images using machine learning algorithms after applying Principal Component Analysis (PCA) for dimensionality reduction. The dataset contains various natural images, and the goal is to accurately predict the class of each image based on the features extracted.

## Methodology
1. **Data Preprocessing**:
   - Loaded the image dataset and performed basic preprocessing steps to ensure data quality.
   - Applied data augmentation techniques to enhance the dataset.

2. **Dimensionality Reduction**:
   - Utilized PCA to reduce the dimensionality of the image data while preserving essential features.
   - Kept 50 principal components to balance performance and computation efficiency.

3. **Model Training**:
   - Split the dataset into training (80%) and testing (20%) sets to evaluate the models effectively.
   - Implemented the following machine learning algorithms:
     - **Logistic Regression**
     - **K-Nearest Neighbors (KNN)**
     - **Random Forest Classifier**

4. **Model Evaluation**:
   - Evaluated each model's performance using accuracy, confusion matrix, and classification report.
   - Visualized the confusion matrices to gain insights into model performance.

## Results
- **Logistic Regression Accuracy**: 0.54
- **K-Nearest Neighbors Accuracy**: 0.49
- **Random Forest Accuracy**: 0.62

The Random Forest model achieved the highest accuracy among the three models tested.

## Technologies Used
- Python
- Libraries: `numpy`, `pandas`, `scikit-learn`, `seaborn`, `matplotlib`

## Usage
1. Clone this repository to your local machine.
2. Install the required libraries (if not already installed).
3. Run the Jupyter Notebook or Python scripts to preprocess the data, train the models, and visualize the results.

## Conclusion
This project demonstrates the application of PCA for dimensionality reduction and the effectiveness of various machine learning algorithms in image classification tasks. Further improvements can be made by fine-tuning the models and exploring additional data augmentation techniques.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
