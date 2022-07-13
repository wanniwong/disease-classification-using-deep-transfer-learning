# Classification of Gastrointestinal Diseases using Deep Transfer Learning

As a part of the digestive system, gastrointestinal (GI) tract is important to involve in digestive activities by breaking down food as consumed into compounds and absorbing nutrients from them. Digestive-related problems such as bloating, constipation, and diarrhoea, should be treated seriously as they could be signs and symptoms of chronic diseases such as cancer. Therefore, endoscope is used for GI examination to allow video or image capture of the GI tract. With the availability of GI endoscopic image data, data scientists and doctors can work together to analyse patient records in order to best understand the GI diseases. This project is aimed to implement deep transfer learning for classifying GI diseases. Initially, the GI dataset with 3 classes and 5542 endoscopic images is collected from two sources – Kvasir and Hyper-Kvasir. After exploratory data analysis, pre-processing steps are taken to clean, standardize, and transform the data. [**data_selection.ipynb**](data selection.ipynb) and data augmentation techniques are then conducted to avoid overfitting during model development. In this project, pretrained ResNet50 which is a convolutional neural network with 50 layers, is proposed for the classification task. To evaluate the model, performance metrics and benchmarking are applied. Based on the obtained results, the proposed model demonstrates high stability with consistent scores. As compared with ResNet18 and ResNet34 models, it also achieves higher accuracy, precision, recall, and F1-score with 94% in average. At the end of the project, ResNet50 is deployed in a web application where users can upload images to generate a prediction dashboard.
