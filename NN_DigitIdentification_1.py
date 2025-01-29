import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from torchvision import datasets, transforms

class SimpleDigitRecognizer(nn.Module):
    def __init__(self):
        super(SimpleDigitRecognizer, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SimpleDigitRecognizer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()
    
    # Basic transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load and split dataset
    full_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    best_val_loss = float('inf')
    patience = 3  # Reduced patience for faster training
    patience_counter = 0
    max_epochs = 8  # Reduced max epochs
    
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        val_loss /= len(val_loader)
        train_accuracy = 100. * correct / total
        val_accuracy = 100. * val_correct / val_total
        
        print(f'Epoch: {epoch}')
        print(f'Training Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
        
        # Save best model and check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'simple_digit_model.pth')
            print("Saved new best model")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    return model

class DrawingApp:
    def __init__(self, root, model):
        self.root = root
        self.root.title("Simple Digit Recognition")
        
        # Canvas setup
        self.canvas = tk.Canvas(root, width=280, height=280, bg='black')
        self.canvas.pack(pady=20)
        
        # Result display
        self.result_text = tk.StringVar()
        self.result_label = tk.Label(root, textvariable=self.result_text)
        self.result_label.pack()
        
        # Clear button
        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()
        
        # Drawing setup
        self.image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.image)
        
        # Model setup
        self.model = model
        self.model.eval()
        
        # Bind events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.recognize_digit)
        
        self.last_x = None
        self.last_y = None

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line((self.last_x, self.last_y, x, y), 
                                  fill='white', width=20)
            self.draw.line([self.last_x, self.last_y, x, y], 
                          fill='white', width=20)
        self.last_x = x
        self.last_y = y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.image)
        self.result_text.set("")
        self.last_x = None
        self.last_y = None

    def recognize_digit(self, event):
        self.last_x = None
        self.last_y = None
        
        # Basic preprocessing
        img = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img)
        img_array = img_array / 255.0
        
        # Make prediction
        with torch.no_grad():
            tensor = torch.FloatTensor(img_array).unsqueeze(0)
            outputs = self.model(tensor)
            probabilities = torch.exp(outputs)[0].numpy()
            
        # Display results
        result = "Predictions:\n"
        for i, prob in enumerate(probabilities):
            result += f"Digit {i}: {prob*100:.1f}%\n"
        self.result_text.set(result)

if __name__ == "__main__":
    try:
        model = SimpleDigitRecognizer()
        model.load_state_dict(torch.load('simple_digit_model.pth'))
        print("Loaded existing model")
    except:
        print("Training new model...")
        model = train_model()
    
    root = tk.Tk()
    app = DrawingApp(root, model)
    root.mainloop()