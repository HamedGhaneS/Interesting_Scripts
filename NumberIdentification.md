# Handwritten Digit Recognition System

A step-by-step explanation of how this digit recognition system works, from training to user interaction.

## System Flow

```
+------------------+     +-----------------+     +------------------+
|                  |     |                 |     |                  |
| Start Program    |---->| Check for Model |---->| Load/Train Model |
|                  |     |                 |     |                  |
+------------------+     +-----------------+     +------------------+
                                                         |
                                                         v
+------------------+     +-----------------+     +------------------+
|                  |     |                 |     |                  |
| Show Prediction  |<----| Process Image   |<----| Draw Digit      |
|                  |     |                 |     |                  |
+------------------+     +-----------------+     +------------------+
         |                                              ^
         |                                              |
         +----------------------------------------------+
```

## Detailed Process Outline

### 1. Model Architecture
   * Input Layer: 784 neurons (representing 28x28 pixel image)
   * First Hidden Layer: 128 neurons with ReLU
   * Second Hidden Layer: 64 neurons with ReLU
   * Output Layer: 10 neurons for digits 0-9
   * Dropout layers: 20% dropout rate

### 2. Training Process
   * Load MNIST dataset
   * Convert images to tensors
   * Normalize data
   * Train for 5 epochs:
     - Process batches of 64 images
     - Calculate loss
     - Update weights
   * Save trained model to file

### 3. Drawing Interface Setup
   * Create 280x280 black canvas
   * Initialize drawing tools
   * Set up clear button
   * Prepare result display area

### 4. Recognition Process
   * Capture user drawing
   * Process image:
     - Resize to 28x28 pixels
     - Convert to grayscale
     - Normalize pixel values
   * Run through neural network
   * Display probability for each digit

### 5. Key Functions
   * **DigitRecognizer**: Neural network model
   * **train_model()**: Handles model training
   * **DrawingApp**: GUI and drawing interface
   * **paint()**: Handles drawing events
   * **recognize_digit()**: Processes and classifies drawings
## Required Libraries
- PyTorch (Neural network)
- Tkinter (GUI)
- PIL (Image processing)
- NumPy (Numerical operations)
- torchvision (MNIST dataset)

## Usage Instructions
1. Run the script
2. If no trained model exists:
   - System will train new model
   - Progress will be displayed
3. Once ready:
   - Draw digit on black canvas
   - Release mouse to see predictions
   - Use clear button to reset
4. View results:
   - Percentage confidence for each digit
   - Higher percentage indicates more likely match

This implementation provides a complete pipeline from training a neural network to providing a user-friendly interface for real-time digit recognition.
