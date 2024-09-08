# 🛒 **FastText-(Meta)-Ecommerce-Category-Classification**

This project demonstrates how to perform text classification on e-commerce product descriptions using FastText.

## 📊 Dataset

The dataset used in this project contains e-commerce item descriptions categorized into four classes:

1. 🏠 Household
2. 🖥️ Electronics
3. 🧥 Clothing and Accessories
4. 📚 Books

Dataset source: [Kaggle - E-commerce Text Classification](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification)

## 🔧 Data Preparation

### Loading the Data

We use pandas to load and inspect the dataset:

```python
import pandas as pd

df = pd.read_csv("ecommerce_dataset.csv", names=["category", "description"], header=None)
print(df.shape)
df.head(3)
```

Output:
```
(50425, 2)
   category                                        description
0  Household  Paper Plane Design Framed Wall Hanging Motivat...
1  Household  SAF 'Floral' Framed Painting (Wood, 30 inch x ...
2  Household  SAF 'UV Textured Modern Art Print Framed' Pain...
```

### Preparing Labels for FastText

FastText expects labels to be prefixed with `__label__`. We create a new column combining the label and description:

```python
df['category'] = '__label__' + df['category'].astype(str)
df['category_description'] = df['category'] + ' ' + df['description']
```

## 🧹 Text Preprocessing

We preprocess the text data using regular expressions to:
1. Remove punctuation
2. Remove extra spaces
3. Convert text to lowercase

```python
import re

def preprocess(text):
    text = re.sub(r'[^\w\s\']',' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip().lower() 

df['category_description'] = df['category_description'].map(preprocess)
```

## 💾 Generating CSV for FastText

We split the data into training and testing sets, then save them as CSV files:

```python
train.to_csv("ecommerce.train", columns=["category_description"], index=False, header=False)
test.to_csv("ecommerce.test", columns=["category_description"], index=False, header=False)
```

## 🏋️ Training and Evaluation

We use FastText to train the model and evaluate its performance:

```python
import fasttext

model = fasttext.train_supervised(input="ecommerce.train")
model.test("ecommerce.test")
```

Results:
```
(10085, 0.9682697074863659, 0.9682697074863659)
```

The model achieves approximately 96.83% precision and recall on the test set.

## 🔮 Predictions

We can use the trained model to make predictions on new product descriptions. Let's examine some examples:

### 🖥️ Electronics Prediction

```python
product_description = "wintech assemble desktop pc cpu 500 gb sata hdd 4 gb ram intel c2d processor 3"
prediction = model.predict(product_description)
print(f"Product: {product_description}")
print(f"Predicted Category: {prediction[0][0]}")
print(f"Confidence: {prediction[1][0]:.2%}")
```

Output:
```
Product: wintech assemble desktop pc cpu 500 gb sata hdd 4 gb ram intel c2d processor 3
Predicted Category: __label__electronics
Confidence: 98.56%
```

The model correctly identifies this as an electronics product with high confidence.

### 🧥 Clothing and Accessories Prediction

```python
product_description = "ockey men's cotton t shirt fabric details 80 cotton 20 polyester super combed cotton rich fabric"
prediction = model.predict(product_description)
print(f"Product: {product_description}")
print(f"Predicted Category: {prediction[0][0]}")
print(f"Confidence: {prediction[1][0]:.2%}")
```

Output:
```
Product: ockey men's cotton t shirt fabric details 80 cotton 20 polyester super combed cotton rich fabric
Predicted Category: __label__clothing_accessories
Confidence: 100.00%
```

The model correctly classifies this as a clothing item with very high confidence.

### 📚 Books Prediction

```python
product_description = "think and grow rich deluxe edition"
prediction = model.predict(product_description)
print(f"Product: {product_description}")
print(f"Predicted Category: {prediction[0][0]}")
print(f"Confidence: {prediction[1][0]:.2%}")
```

Output:
```
Product: think and grow rich deluxe edition
Predicted Category: __label__books
Confidence: 100.00%
```

The model accurately identifies this as a book with very high confidence.

## 🔍 Word Similarities

We can also find similar words using the trained model:

```python
model.get_nearest_neighbors("painting")
```

Output:
```
[(0.9976388216018677, 'vacuum'),
 (0.9968333840370178, 'guard'),
 (0.9968314170837402, 'heating'),
 (0.9966275095939636, 'lid'),
 (0.9962871670722961, 'lamp'),
 ...]
```

This shows words that the model considers similar to "painting" in the context of e-commerce products.

```python
model.get_nearest_neighbors("sony")
```

Output:
```
[(0.9988397359848022, 'external'),
 (0.998672366142273, 'binoculars'),
 (0.9981507658958435, 'dvd'),
 (0.9975149631500244, 'nikon'),
 (0.9973592162132263, 'glossy'),
 ...]
```

These results show words that the model associates closely with the brand "Sony" in the e-commerce context.

## 🚀 Conclusion

This project demonstrates the effectiveness of FastText in classifying e-commerce product descriptions. With high accuracy and the ability to make quick predictions, this model can be a valuable tool for automating product categorization in e-commerce platforms.

For further improvements, consider:
- Experimenting with different preprocessing techniques
- Fine-tuning FastText hyperparameters
- Exploring other deep learning models for comparison

Happy classifying! 🎉
