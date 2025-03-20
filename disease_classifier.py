import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Step 1: Load and preprocess the data
def load_and_preprocess_data(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Extract features and target
    # Exclude patient_id and diagnosis columns from features
    X = df.drop(['patient_id', 'diagnosis'], axis=1)
    y = df['diagnosis']
    
    # Convert sex to numeric (M=0, F=1)
    X['sex'] = X['sex'].map({'M': 0, 'F': 1})
    
    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale the numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder.classes_

# Step 2: Create a PyTorch Dataset
class DiseaseDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Step 3: Define the Neural Network model
class DiseaseClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(DiseaseClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x  # Using CrossEntropyLoss, so no need for softmax here
    
    def predict(self, x):
        with torch.no_grad():
            x = self.forward(x)
            return self.softmax(x)

# Step 4: Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10000):
    # Lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_acc = 100 * val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)
        
        # Print statistics every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, '
                  f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, '
                  f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# Step 5: Evaluate the model
def evaluate_model(model, test_loader, criterion, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate test loss and accuracy
    test_loss /= len(test_loader)
    test_acc = accuracy_score(all_labels, all_preds) * 100
    
    # Print evaluation metrics
    print(f'\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Create and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    return test_loss, test_acc, all_preds, all_labels

# Step 6: Function to plot training and validation metrics
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Step 7: Function to make predictions on new data
def predict_disease(model, scaler, new_data, class_names):
    # Preprocess the new data similar to the training data
    if isinstance(new_data, pd.DataFrame):
        # If it's a DataFrame, convert sex to numeric
        if 'sex' in new_data.columns:
            new_data['sex'] = new_data['sex'].map({'M': 0, 'F': 1})
        
        # Scale the features
        new_data_scaled = scaler.transform(new_data)
    else:
        # If it's already a numpy array
        new_data_scaled = scaler.transform(new_data.reshape(1, -1))
    
    # Convert to tensor
    new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model.predict(new_data_tensor)
        prob, pred_class = torch.max(outputs, 1)
    
    # Return predicted class and probabilities
    predicted_disease = class_names[pred_class.item()]
    probabilities = {class_name: float(prob) for class_name, prob in zip(class_names, outputs[0])}
    
    return predicted_disease, probabilities

# Main execution flow
def main(file_path='disease_dataset.csv'):
    # Step 1: Load and preprocess data
    X_train, X_test, y_train, y_test, class_names = load_and_preprocess_data(file_path)
    print(f"Classes: {class_names}")
    print(f"Training samples: {len(y_train)}, Testing samples: {len(y_test)}")
    
    # Step 2: Create datasets and dataloaders
    train_dataset = DiseaseDataset(X_train, y_train)
    test_dataset = DiseaseDataset(X_test, y_test)
    
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Step 3: Initialize the model
    input_size = X_train.shape[1]  # Number of features
    hidden_size = 64  # Size of hidden layer
    num_classes = len(class_names)  # Number of disease classes
    
    model = DiseaseClassifier(input_size, hidden_size, num_classes)
    
    # Step 4: Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Step 5: Train the model
    print("Starting training...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs=100
    )
    
    # Step 6: Evaluate the model
    print("\nEvaluating model on test data...")
    test_loss, test_acc, all_preds, all_labels = evaluate_model(model, test_loader, criterion, class_names)
    
    # Step 7: Plot metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Optional: Save the model
    torch.save(model.state_dict(), 'disease_classifier_model.pth')
    print("Model saved as 'disease_classifier_model.pth'")
    
    return model, X_train, X_test, y_train, y_test, class_names

# Example usage
if __name__ == "__main__":
    # Run the main execution flow
    model, X_train, X_test, y_train, y_test, class_names = main('./disease_dataset.csv')
    
    # Example: Make a prediction for a new patient
    # Create sample data (this would be replaced with real patient data)
    sample_patient = {
        'age': 25, 'sex': 'M', 'fever': 0, 'sore_throat': 0, 'cough': 1, 
        'headache': 1, 'fatigue': 1, 'body_ache': 0, 'runny_nose': 0, 
        'congestion': 0, 'shortness_of_breath': 0, 'nausea': 0, 
        'vomiting': 0, 'diarrhea': 0, 'chills': 0, 'rash': 0, 
        'chest_pain': 0, 'dizziness': 0, 'swollen_lymph_nodes': 0, 
        'loss_of_appetite': 0, 'joint_pain': 0, 'abdominal_pain': 0, 
        'ear_pain': 0, 'eye_redness': 0, 'loss_of_taste': 0, 
        'loss_of_smell': 0, 'wheezing': 0, 'hoarse_voice': 0, 
        'difficulty_swallowing': 0, 'muscle_weakness': 0, 
        'night_sweats': 0, 'confusion': 0, 'rapid_breathing': 0, 
        'jaundice': 0, 'itching': 0, 'bruising': 0, 'blood_in_stool': 0, 
        'weight_loss': 0, 'insomnia': 0, 'sweating': 0, 
        'symptom_duration_days': 3
    }
    
    # Convert to DataFrame
    sample_df = pd.DataFrame([sample_patient])
    
    # Get trained scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Make prediction
    predicted_disease, probabilities = predict_disease(model, scaler, sample_df, class_names)
    print(f"\nPredicted disease: {predicted_disease}")
    print("\nProbabilities:")
    for disease, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        print(f"{disease}: {prob:.4f}")