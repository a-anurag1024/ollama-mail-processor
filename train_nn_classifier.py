import torch

from mail_classifier.nn_trainer import train
from mail_classifier.nn_classifier import MailClassifier


classifier_model = MailClassifier(cache_folder="./mount/models", 
                                  category_classes=7, 
                                  action_classes=3)

class_labels = {
    "category": {
        "Education": 0,
        "Newsletters": 1,
        "Personal": 2,
        "Promotions": 3,
        "Social": 4,
        "Work": 5,
        "Unknown": 6
    },
    
    "action": {
        "READ": 0,
        "IGNORE": 1,
        "ACT": 2
    }
}


train(
    training_data_file="./mount/train_data.csv",
    class_labels=class_labels,
    mail_classifier=classifier_model,
    BATCH_SIZE=32,
    NUM_EPOCHS=50,
    LEARNING_RATE=1e-4,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    checkpoint_dir="./mount/model_checkpoints_1"
)