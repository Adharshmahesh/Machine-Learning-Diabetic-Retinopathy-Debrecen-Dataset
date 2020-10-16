# Machine-Learning-Diabetic-Retinopathy-Debrecen-Dataset
This is a binary classification task. This task is implemented using standard Machine learning models - Logistic Regression and Naive Bayes. These models are implemented from scratch using Python and using the libraries in Scikit learn. 

# Dataset Information
This dataset contains features extracted from the Messidor image set to predict whether an image contains signs of diabetic retinopathy or not. All features represent either a detected lesion, a descriptive feature of a anatomical part or an image-level descriptor. The underlying method image analysis and feature extraction as well as our classification technique is described in Balint Antal, Andras Hajdu: An ensemble-based system for automatic screening of diabetic retinopathy, Knowledge-Based Systems 60 (April 2014), 20-27.

# Attribute Information
0) The binary result of quality assessment. 0 = bad quality 1 = sufficient quality.
1) The binary result of pre-screening, where 1 indicates severe retinal abnormality and 0 its lack.
2-7) The results of MA detection. Each feature value stand for the
number of MAs found at the confidence levels alpha = 0.5, . . . , 1, respectively.
8-15) contain the same information as 2-7) for exudates. However,
as exudates are represented by a set of points rather than the number of
pixels constructing the lesions, these features are normalized by dividing the
number of lesions with the diameter of the ROI to compensate different image
sizes.
16) The euclidean distance of the center of
the macula and the center of the optic disc to provide important information
regarding the patientâ€™s condition. This feature
is also normalized with the diameter of the ROI.
17) The diameter of the optic disc.
18) The binary result of the AM/FM-based classification.
19) Class label. 1 = contains signs of DR (Accumulative label for the Messidor classes 1, 2, 3), 0 = no signs of DR.

# Dataset
Dataset can be downloaded from UCI Machine Learning Repository https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set
