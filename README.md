# CS430 Group Project

## Usage

### Requirements
This program relies on the requirements specified in **requirements.txt**.\
Use the command `pip install -r requirements.txt` to install the requrements.

### MLALGS.py
This file is used to generate the convergence graphs for the Artificial Neural Network and the Stochastic Gradient Descent Linear Regression models.\
It has three variables at the top of the file to specify which variation of training data to use, and which test case to use.\
**VARIANT** Specifies the training variant (1-5)\
**USE_ENTIRE_TEST** When **True**, specifies that the entire training dataset should be used (10k examples), and when **False** specifies that an individual test case should be used\
**TEST_CASE** Specifies the test case to use if **USE_ENTIRE_TEST** is set to **False**, values (1-10)

### MLALGS_Applied.py
This file is used to generate the performance metrics and graph them.\
These metrics are colleced across all 5 variations of training data, and each variation is applied to all 10 test cases.\
The average of the metrics across the test cases are used for plotting **Hits@10**, **AUC@10**, **MAE**, and **Eval Time**.\
The **PRINT_DETAILS** variable at the top of the file controls if the models will print extra information during their training and evaluation.

## Project Structure
```
.
├── MLALGS_Applied.py
├── MLAlgs.py
├── README.md
├── facebook_comment_volume_dataset
│   ├── Dataset
│   │   ├── Catagory_File - Feature 4.pdf
│   │   ├── Testing
│   │   │   ├── Features_TestSet.csv
│   │   │   └── TestSet
│   │   │       ├── Test_Case_1.arff
│   │   │       ├── Test_Case_1.csv
│   │   │       ├── Test_Case_10.arff
│   │   │       ├── Test_Case_10.csv
│   │   │       ├── Test_Case_2.arff
│   │   │       ├── Test_Case_2.csv
│   │   │       ├── Test_Case_3.arff
│   │   │       ├── Test_Case_3.csv
│   │   │       ├── Test_Case_4.arff
│   │   │       ├── Test_Case_4.csv
│   │   │       ├── Test_Case_5.arff
│   │   │       ├── Test_Case_5.csv
│   │   │       ├── Test_Case_6.arff
│   │   │       ├── Test_Case_6.csv
│   │   │       ├── Test_Case_7.arff
│   │   │       ├── Test_Case_7.csv
│   │   │       ├── Test_Case_8.arff
│   │   │       ├── Test_Case_8.csv
│   │   │       ├── Test_Case_9.arff
│   │   │       └── Test_Case_9.csv
│   │   └── Training
│   │       ├── Features_Variant_1.arff
│   │       ├── Features_Variant_1.csv
│   │       ├── Features_Variant_2.arff
│   │       ├── Features_Variant_2.csv
│   │       ├── Features_Variant_3.arff
│   │       ├── Features_Variant_3.csv
│   │       ├── Features_Variant_4.arff
│   │       ├── Features_Variant_4.csv
│   │       ├── Features_Variant_5.arff
│   │       └── Features_Variant_5.csv
│   └── __MACOSX
│       └── Dataset
│           ├── Testing
│           │   └── TestSet
│           └── Training
└── requirements.txt
```