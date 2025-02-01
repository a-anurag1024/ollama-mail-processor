from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn
import os
import glob
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd

from mail_classifier.nn_classifier import MailDataset, mail_collate_fn


def train(training_data_file, 
          class_labels,
          mail_classifier,
          BATCH_SIZE=16,
          NUM_EPOCHS=100,
          LEARNING_RATE=1e-4,
          device=torch.device('cuda:0'),
          checkpoint_dir="checkpoints"):
    """
    Train function for the MailClassifier model with TensorBoard logging and checkpointing.

    Parameters:
    -----------
    training_data_file (str): Path to the CSV file containing training data.
    class_labels (dict): Mapping of category and action labels to integers.
    mail_classifier (nn.Module): The PyTorch model to train.
    BATCH_SIZE (int): Batch size for DataLoader.
    NUM_EPOCHS (int): Number of epochs to train.
    LEARNING_RATE (float): Learning rate for the optimization.
    device (torch.device): Device for training (CPU or GPU).
    checkpoint_dir (str): Directory to save model checkpoints.
    """
    # Prepare directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, "logs"))
    
    # Load and split data
    df = pd.read_csv(training_data_file)
    data = df.to_dict(orient="records")
    train_data, validation_data = train_test_split(data, test_size=0.1, random_state=42)
    train_dataset = MailDataset(train_data, class_labels=class_labels)
    val_dataset = MailDataset(validation_data, class_labels=class_labels)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=mail_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=mail_collate_fn)
    
    # Send model to device
    mail_classifier.to(device)
    
    # Loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    total_params = sum(p.numel() for p in mail_classifier.parameters() if p.requires_grad)
    print(f"TOTAL PARAMETERS GETTING TRAINED: {total_params}")
    optimizer = torch.optim.Adam(mail_classifier.parameters(), lr=LEARNING_RATE)
    
    # Store the best models
    best_models = []
    
    for epoch in range(1, NUM_EPOCHS + 1):
        mail_classifier.train()  # Training mode
        train_loss = 0.0
        
        # Training step
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}/{NUM_EPOCHS} - Training"):
            X, Y_category, Y_action = batch
            Y_category, Y_action = Y_category.to(device), Y_action.to(device)
            
            # Forward pass
            category_logits, action_logits = mail_classifier(list(X))
            category_loss = loss_function(category_logits, Y_category)
            action_loss = loss_function(action_logits, Y_action)
            loss = category_loss + action_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dataloader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        print(f"Epoch {epoch}: Avg Train Loss = {avg_train_loss:.4f}")
        
        # Validation step
        mail_classifier.eval()  # Evaluation mode
        val_loss = 0.0
        correct_category, correct_action, total = 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch}/{NUM_EPOCHS} - Validation"):
                X, Y_category, Y_action = batch
                X = list(X)
                Y_category, Y_action = Y_category.to(device), Y_action.to(device)
                
                # Forward pass
                category_logits, action_logits = mail_classifier(X, add_random_noise=True)
                
                # Compute losses
                category_loss = loss_function(category_logits, Y_category)
                action_loss = loss_function(action_logits, Y_action)
                loss = category_loss + action_loss
                val_loss += loss.item()
                
                # Compute accuracy
                category_preds = torch.argmax(category_logits, dim=1)
                action_preds = torch.argmax(action_logits, dim=1)
                correct_category += (category_preds == Y_category).sum().item()
                correct_action += (action_preds == Y_action).sum().item()
                total += len(Y_category)
        
        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = (correct_category + correct_action) / (2 * total) * 100
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)
        print(f"Epoch {epoch}: Avg Val Loss = {avg_val_loss:.4f}, Val Accuracy = {val_accuracy:.2f}%")
        
        # Save the model checkpoint if it's among the top 5
        checkpoint_file = os.path.join(checkpoint_dir, f"epoch-{epoch:03d}_val-loss-{avg_val_loss:.4f}.pt")
        torch.save(mail_classifier.state_dict(), checkpoint_file)
        saved_files = glob.glob(f"{checkpoint_dir}/epoch-*.pt")
        saved_models = [(float(os.path.basename(p).split('val-loss-')[1].split('.pt')[0]), p) for p in saved_files]
        saved_models.append((avg_val_loss, checkpoint_file))
        saved_models = sorted(saved_models, key=lambda x: x[0])
        if len(saved_models) > 5: 
            delete_models = [v[1] for v in saved_models[5:]]    # keep top 5 models
            for p in delete_models:
                os.remove(p)
    
    writer.close()
