# Emotion Detection using FER Dataset

This project implements an **emotion detection model** using the [FER (Facial Expression Recognition) dataset](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer) from Kaggle. The model classifies facial images into different emotions using a deep learning approach.

---

## Dataset

- The dataset is available on [Kaggle](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer).  
- It is divided into `train` and `test` sets.  
- **Important:** The original dataset directories need to be used in place of the `train/` and `test/` folders in the app.

---

## Project Setup

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd <your-repo>
```

2. **Set up virtual environment**
```bash
    python -m venv env
    env\Scripts\activate  # On Windows
    # source env/bin/activate  # On Linux / macOS
```
3. **Install dependencies**
 ```bash  
pip install -r requirements.txt
```
DataSets/
├── train/
├── test/


4. **Training**

The model uses early stopping, so it will automatically stop training when the validation loss stops improving.

You can modify the number of epochs as needed.

Example training snippet:

model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=[early_stopping]
)


## Notes

The train/ and test/ folders are ignored in Git due to large data size. Please ensure the dataset is downloaded manually.

Model weights are saved automatically in .h5 format and are also ignored in Git.

.gitignore includes:

env/
.ipynb_checkpoints/
anaconda_projects/
*.h5
requirements.txt
train/
test/

## License
```bash
This project uses the FER dataset from Kaggle under the dataset's licensing terms.

This version has **setup, training, evaluation, and inference** included, ready for anyone to follow.
```
