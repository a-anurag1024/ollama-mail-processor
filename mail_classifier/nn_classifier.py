import torch
from torch.utils.data import Dataset
from torch import nn
from sentence_transformers import SentenceTransformer




class MailDataset(Dataset):
    def __init__(self, 
                 data,
                 class_labels):
        """
        Torch dataset builder for the mails data
        
        Parameters:
        ```````````
        data (List): list of the mail objects
        class_labels (dict): integer labels for the different classes of category and action
        """
        self.mails = data
        self.class_labels = class_labels
    
    
    def __len__(self):
        return len(self.mails)
    
    
    def __getitem__(self, index):
        mail = self.mails[index]
        return {"subject": mail['subject'],
                'category': torch.tensor(self.class_labels['category'][mail['category']]),
                'action': torch.tensor(self.class_labels['action'][mail['action']])}
        """
        return (mail['subject'], 
                torch.tensor(self.class_labels['category'][mail['category']]), 
                torch.tensor(self.class_labels['action'][mail['action']]))
        """
    
    
def mail_collate_fn(batch):
    """
    Custom collate function to process batches of data for the MailDataset.
    
    Parameters:
    ```````````
    batch (List): List of tuples returned by the dataset's __getitem__ method
    
    Returns:
    ````````
    - tokenized_subjects: List[str] or tensor if tokenized
    - categories: torch.Tensor of category labels
    - actions: torch.Tensor of action labels
    """
    subjects = [item['subject'] for item in batch]
    categories = torch.stack([item['category'] for item in batch])
    actions = torch.stack([item['action'] for item in batch])
    
    return subjects, categories, actions


class MailClassifier(nn.Module):
    def __init__(self, cache_folder=None, category_classes=7, action_classes=3):
        """ 
        Pytorch Model Class for the Mail Classification
        
        Parameters:
        ```````````
        cache_folder (str|pathlike): path to the sentence transformer model weights cache_folder
        category_classes (int): number of classes in category
        action_classes (int): number of classes in action
        """
        super(MailClassifier, self).__init__()
        self.st_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_folder)
        
        # Freeze the SentenceTransformer layers
        for param in self.st_model.parameters():
            param.requires_grad = False
            
        self.common_layer_1 = nn.Linear(384, 192)
        self.common_layer_2 = nn.Linear(192, 96)
        self.common_layer_3 = nn.Linear(96, 48)
        self.category_classifier = nn.Sequential(
            nn.Linear(48, 24),
            nn.Linear(24, 12),
            nn.Linear(12, category_classes)
        )
        self.action_classifier = nn.Sequential(
            nn.Linear(48, 24),
            nn.Linear(24, 12),
            nn.Linear(12, action_classes)
        )
    
    
    def forward(self, mail_subjects, add_random_noise=False):
        """
        Parameters:
        -----------
        mail_subjects (list[str]): List of mail subjects.
        add_random_noise (bool): whether to add some random noise to the text embeddings
                                useful during training. Default is False
        
        Returns:
        --------
        category_logits (torch.Tensor): Logits for category classification.
        action_logits (torch.Tensor): Logits for action classification.
        """
        embeddings = self.st_model.encode(mail_subjects, convert_to_tensor=True)
        if add_random_noise:
            embeddings += torch.randn_like(embeddings) * 0.00 * torch.norm(embeddings)

        x = self.common_layer_1(embeddings)
        x = torch.relu(x)
        x = self.common_layer_2(x)
        x = torch.relu(x)
        x = self.common_layer_3(x)
        x = torch.relu(x)
        
        category_logits = self.category_classifier(x)
        action_logits = self.action_classifier(x)
        
        return category_logits, action_logits
    
    
    def predict_category(self, mail_subjects):
        """
        Predict the category class for given mail subjects.

        Parameters:
        -----------
        mail_subjects (list[str]): List of mail subjects.
        
        Returns:
        --------
        category_predictions (list[int]): Predicted category class indices.
        """
        self.eval()
        with torch.no_grad():
            category_logits, _ = self.forward(mail_subjects)
            category_predictions = torch.argmax(category_logits, dim=1).tolist()
        return category_predictions


    def predict_action(self, mail_subjects):
        """
        Predict the action class for given mail subjects.

        Parameters:
        -----------
        mail_subjects (list[str]): List of mail subjects.
        
        Returns:
        --------
        action_predictions (list[int]): Predicted action class indices.
        """
        self.eval()
        with torch.no_grad():
            _, action_logits = self.forward(mail_subjects)
            action_predictions = torch.argmax(action_logits, dim=1).tolist()
        return action_predictions
