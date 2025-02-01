# ollama-mail-processor and student-mail-multiclassifier
Processing and Retrieving useful information from collected e-mails using Ollama and langchain. And then using the information to train smaller classifier. 
The multi-classifier trained using this simple knowledge distillation pipeline will be used to predict the type of e-mail and the suggested action for the e-mail looking at the subject of the e-mail. 


# Scripts :-
1. `run_classifier.py` : run the classifier using LLAMA3.1 model (via ollama)
2. `upload_data_to_local_db.py` : upload the data to a local MySQL database (hosted using docker)
3. `train_nn_classifier.py` : train a student multiclassifier using the classification done by the LLAMA model.
4. `notebooks/classifier_data_prep.ipynb` : notebook to retrieve and preprocess data from the local db

# Student Multiclassifier model
The student classifier is built by using the **all-MiniLM-L6-v2** sentence transformer as the initial layers of the model which is followed by a few layers of common ANN. The common ANN layers are then branched to two different smaller nets, one for each classifier. In total, the whole classifier is 22M parameters but only 100k parameters are trainable and other parameters have been freezed. 


The two sub-classifiers are:
1. Mail Category Predictor: To classify the mail to one of these categories ("Education", "Newsletters", "Personal", "Promotions", "Social", "Work", "Unknown")
2. Recommended Action Predictor: To predict what action out of "READ", "IGNORE" or "ACT" is suggested for the mail.


The Multiclassifier was trained for 50 epochs (on top of 10k mails), and achieved the minimum validation loss after around 28 epochs. Despite highly skewed dataset (of a single personal gmail account with very less variations and volume), the training got a respectable accuracy of 85% on the recommended action prediction and 75% on the category prediction.


To further improve performance, some data augmentation can be done using ollama to further diversify the data and increase its volume.