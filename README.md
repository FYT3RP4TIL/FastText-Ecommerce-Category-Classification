# **FastText_Meta-Ecommerce-Category-Classification**
### 🛍️ **Project Overview**
This project leverages the powerful FastText algorithm to classify ecommerce products into their respective categories based on their descriptions. By combining the strength of natural language processing with machine learning, we've created a robust system that accurately categorizes products, enhancing user experience and improving product discoverability.

### 📊 **Dataset**
* **Source:** [Kaggle-Ecommerce-Text-Classification](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification)
* **Format:** CSV
* **Columns:**
    * `category`: Product category
    * `description`: Product description
* **Categories:** Household, Electronics, Clothing and Accessories, Books

### 🚀 **Data Preprocessing**
1. **Data Cleaning:** 🧹 Removing null values and handling missing data to ensure data integrity.
2. **Label Formatting:** 🔖 Prefixing the category with "__label__" to conform to FastText's label format.
3. **Category-Description Concatenation:** 🔗 Combining the category and description into a single field for training.
4. **Text Cleaning:** ✂️ Removing punctuation, special characters, and extra spaces using regular expressions to focus on relevant textual information.

### 🧠 **Model Training**
1. **Data Split:** 🔪 Dividing the dataset into training and testing sets (80% for training, 20% for testing).
2. **FastText Model Creation:** ⚙️ Training a FastText supervised model on the training set to learn the underlying patterns and relationships between product descriptions and their corresponding categories.

### 📊 **Model Evaluation**
1. **Testing:** 🧪 Evaluating the model's performance on the testing set to assess its generalization ability.
2. **Metrics:** 📈 Reporting accuracy, precision, recall, and F1-score to provide a comprehensive evaluation of the model's effectiveness.

### 🛠️ **Usage**
1. **Data Preparation:** 📋 Ensure the dataset is in CSV format with the correct column names.
2. **Training:** 🏋️‍♂️ Run the training script to create the FastText model.
3. **Testing:** 🔬 Evaluate the model's performance on the testing set.
4. **Prediction:** 🔮 Use the trained model to predict categories for new product descriptions.

### 💡 **Additional Notes**
* **Hyperparameter Tuning:** 🔧 Consider exploring different hyperparameters (e.g., word embeddings dimension, epochs) to optimize model performance.
* **Visualization:** 📊 Explore techniques like word embedding visualization to gain insights into the model's learned representations.
* **Scalability:** 📈 If the dataset is large, consider using distributed training or GPU acceleration to speed up training.
