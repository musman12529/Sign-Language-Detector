# Sign Language Detector

This project utilizes Google Teachable Machines for training image data to detect and display sign language gestures using Python. The repository contains scripts for capturing training data, training the model, and real-time sign language detection.

## Video Demonstration



https://github.com/musman12529/Sign-Language-Detector/assets/114633620/b4d072ba-b542-4059-83c4-a531d9df7820



## Requirements

- Python 3.x
- cvzone (`pip install cvzone`)
- TensorFlow (if not already installed)

## Usage

### Training Data Collection (`data.py`)

1. Run `data.py` to capture training images.
2. Press 's' to save the captured image with the current timestamp in the specified folder.

### Model Testing (`tester.py`)

1. Run `tester.py` to perform real-time sign language detection.
2. The detected sign language gesture will be displayed on the screen along with a bounding box.

## File Structure

- **Images**: Contains saved images used for training the model.
- **data.py**: Script for capturing training data.
- **tester.py**: Script for real-time sign language detection.
- **keras_model.h5**: Pre-trained Keras model for sign language classification.
- **labels.txt**: Text file containing labels for each class.

## Acknowledgments

- This project utilizes the [cvzone](https://github.com/cvzone/cvzone) library for hand tracking and classification modules.
- Training data for the model was collected using the [Google Teachable Machines](https://teachablemachine.withgoogle.com/) platform.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request to improve this project.


