# SMS-Spam-Collection

## **Objective**  
The **SMS Spam Collection v.1** is a dataset designed for research in SMS spam filtering. It consists of **5,574 SMS messages**, categorized as **ham (legitimate)** or **spam**. The goal of this dataset is to facilitate the development and evaluation of machine learning models for spam detection.

---

## **Data Description**  
The dataset is composed of SMS messages collected from various sources and labeled as **ham** or **spam**. The messages are primarily in **English** and originate from different geographical regions. The dataset is widely used for **text classification** and **natural language processing (NLP) tasks**.

### **Sources of Data**  
The dataset was compiled from multiple free or publicly available research sources, including:
- **Grumbletext Web Forum**: 425 manually extracted SMS spam messages.
- **Caroline Tag's PhD Thesis**: 450 legitimate SMS messages.
- **NUS SMS Corpus (NSC)**: 3,375 legitimate SMS messages from the National University of Singapore.
- **SMS Spam Corpus v.0.1 Big**: 1,002 ham messages and 322 spam messages.

### **Dataset Statistics**  
- **Total Messages**: 5,574  
- **Ham (Legitimate) Messages**: 4,827 (86.6%)  
- **Spam Messages**: 747 (13.4%)  

---

## **Data Format**  
The dataset is provided as a **text file** where each line contains one SMS message. Each line consists of two columns:  
1. **Label** – Indicates whether the message is **ham** (legitimate) or **spam**.  
2. **Message Text** – The raw SMS text.  

### **Example Data**  
```
ham    What you doing? How are you?
ham    Ok lar... Joking wif u oni...
spam   FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time!
spam   URGENT! Your Mobile No 07808726822 was awarded a £2,000 Bonus Prize!
```
⚠️ **Note:** Messages are **not sorted chronologically**.

---

## **Steps to Use the Dataset**
1. **Preprocess the Data**  
   - Remove **punctuation** and **special characters**.  
   - Convert text to **lowercase**.  
   - Perform **tokenization and stemming/lemmatization**.  

2. **Feature Engineering**  
   - Convert text to numerical format using **TF-IDF** or **word embeddings**.  
   - Extract key **spam indicators** like the presence of promotional keywords, phone numbers, or URLs.  

3. **Train Machine Learning Models**  
   - Use models like **Naïve Bayes, Support Vector Machines (SVM), Random Forest, or Neural Networks** to classify SMS messages.  

4. **Evaluate Performance**  
   - Use **accuracy, precision, recall, and F1-score** to measure the effectiveness of the spam detection model.  

---

## **Usage**
### **1. Clone the Repository**
```bash
git clone https://github.com/your-repo/sms-spam-collection.git
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Load and Explore the Dataset**
You can load the dataset using **Pandas** in Python:
```python
import pandas as pd

df = pd.read_csv("smsspamcollection.txt", sep="\t", header=None, names=["Label", "Message"])
print(df.head())
```

---

## **Research & References**
This dataset has been used in academic research. For an in-depth study, refer to:
**[1] Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A.**  
*"Contributions to the study of SMS Spam Filtering: New Collection and Results."*  
Proceedings of the 2011 ACM Symposium on Document Engineering (ACM DOCENG'11).

📌 **Original Dataset Repository:** [SMS Spam Collection](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/)  

---

## **License & Disclaimer**
This dataset is **freely available for research purposes**, subject to the following conditions:
- If you use this dataset, please **cite the original research paper**.
- The dataset is provided **"as is"**, with **no warranty** of accuracy or performance.
- The copyright is held by **Tiago Agostinho de Almeida** and **José María Gómez Hidalgo**.

For inquiries, please contact **tiago@dt.fee.unicamp.br**.

---

This format follows the **structured approach** of the Airline Data Clustering Project while keeping all **relevant details from the original README**. Let me know if you need modifications! 🚀
