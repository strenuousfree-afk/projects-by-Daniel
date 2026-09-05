# Leaf vs Non-Leaf Image Classification

## Project Overview

This project develops a binary image classification system that determines whether an input image primarily represents a leaf or a non-leaf object.

The project uses Transfer Learning with MobileNetV2 followed by fine-tuning.

### Classes

- leaf
- non_leaf

---

## Project Objective

The objective is to build a complete image classification pipeline that can:

1. Inspect and clean an image dataset.
2. Separate images into leaf and non-leaf classes.
3. Create training, validation, and test datasets.
4. Train a MobileNetV2-based classifier.
5. Fine-tune the model.
6. Evaluate the model on an independent test set.
7. Save a final deployment model.
8. Predict the class of a new image.

---

## Dataset

The original dataset is stored in:

    dataset/

The raw dataset was preserved throughout the project.

A separate filtered dataset was created for binary classification.

### Cleaned Dataset

| Class | Images |
|---|---:|
| Leaf | 4,389 |
| Non-leaf | 997 |
| Total | 5,386 |

The cleaned dataset was split into:

- Training: 80%
- Validation: 10%
- Test: 10%

The independent test set contains 541 images.

---

## Data Cleaning

The cleaning process included:

- Dataset inspection
- Image preview generation
- Leaf/non-leaf filtering
- Corrupted-image checking
- Image dimension validation
- Duplicate detection
- Duplicate removal

Before duplicate removal:

- Total images: 5,568
- Duplicate groups: 160
- Duplicate copies: 182

After duplicate removal:

- Leaf images: 4,389
- Non-leaf images: 997
- Total images: 5,386
- Remaining duplicate groups: 0

The raw dataset was not modified.

---

## Image Preprocessing

Images were processed using the following pipeline:

1. Convert image to RGB.
2. Resize to 224 x 224.
3. Convert to float32.
4. Normalize pixel values by dividing by 255.
5. Add a batch dimension.

---

## Model

The project uses MobileNetV2 with ImageNet pretrained weights.

Architecture:

    Input Image
         |
         v
    224 x 224 x 3
         |
         v
    MobileNetV2
         |
         v
    Global Average Pooling
         |
         v
    Dropout (0.30)
         |
         v
    Dense (1 neuron)
         |
         v
    Sigmoid
         |
         v
    Leaf / Non-leaf

The MobileNetV2 base was initially frozen while the classification head was trained.

---

## Initial Training

Initial training used:

- Model: MobileNetV2
- Input size: 224 x 224 x 3
- Batch size: 32
- Learning rate: 0.0001
- Epochs: 15
- ImageNet pretrained weights
- Frozen MobileNetV2 base
- Binary classification
- Sigmoid output

The best initial model was saved as:

    models/best_mobilenetv2.keras

---

## Initial Evaluation

The original model was evaluated on the independent 541-image test set.

Results:

- Correct predictions: 535
- Incorrect predictions: 6
- Accuracy: approximately 98.89%

The six misclassified images were retained for error analysis.

---

## Fine-Tuning

Selected deeper MobileNetV2 layers were unfrozen for fine-tuning.

Fine-tuning used:

- Learning rate: 0.00001
- Batch size: 32
- Epochs: 5
- Selected MobileNetV2 layers unfrozen
- Training data for optimization
- Validation data for monitoring
- Independent test set kept separate

The best fine-tuned model was saved as:

    models/fine_tuned/best_finetuned_mobilenetv2.keras

---

## Final Evaluation

The fine-tuned model was evaluated on the same independent 541-image test set.

Results:

- Correct predictions: 540
- Incorrect predictions: 1
- Accuracy: approximately 99.82%

### Model Comparison

| Model | Test Images | Correct | Incorrect | Accuracy |
|---|---:|---:|---:|---:|
| Original MobileNetV2 | 541 | 535 | 6 | ~98.89% |
| Fine-tuned MobileNetV2 | 541 | 540 | 1 | ~99.82% |

The fine-tuned model was selected as the final model.

---

## Final Deployment Model

The final deployment model is:

    models/deployment/final_leaf_non_leaf_model.keras

The deployment model was verified against the selected fine-tuned model.

---

## Single-Image Prediction

A separate image was tested using the final deployment model.

Test image:

    new_test_image.jpg.jpg

Prediction:

- Leaf probability: 0.00%
- Non-leaf probability: 100.00%
- Predicted class: NON_LEAF
- Confidence: 100.00%

The raw model output was approximately:

    0.9999995

The prediction pipeline successfully loaded, resized, normalized, and classified the new image.

---

## Project Structure

    project1/
    |
    +-- dataset/
    |
    +-- filtered_dataset/
    |   +-- leaf/
    |   +-- non_leaf/
    |
    +-- split_dataset/
    |   +-- train/
    |   +-- validation/
    |   +-- test/
    |
    +-- models/
    |   +-- best_mobilenetv2.keras
    |   +-- fine_tuned/
    |   |   +-- best_finetuned_mobilenetv2.keras
    |   +-- deployment/
    |       +-- final_leaf_non_leaf_model.keras
    |
    +-- preview/
    |   +-- error_analysis/
    |
    +-- reports/
    |   +-- final/
    |
    +-- dataset_manifest.csv
    |
    +-- README.md

---

## Important Reports

Important reports include:

    reports/final/step47I/original_vs_finetuned_report.txt

    reports/final/step47K/final_model_selection_report.txt

    reports/removed_duplicates_step35.csv

---

## Prediction Logic

The model produces a sigmoid output.

The output represents the probability of the non_leaf class.

Using a threshold of 0.50:

    prediction >= 0.50  ->  non_leaf

    prediction < 0.50   ->  leaf

---

## Limitations

The model has several limitations:

- The dataset contains more leaf images than non-leaf images.
- Some images can contain both leaves and other objects.
- Performance may decrease on images very different from the training dataset.
- High confidence does not always guarantee a correct prediction.

---

## Future Improvements

Possible improvements include:

- Adding more diverse non-leaf images.
- Adding more real-world images.
- Testing additional pretrained CNN architectures.
- Performing additional error analysis.
- Improving data augmentation.
- Building a web or graphical prediction interface.
- Deploying the model as an API.
- Testing the model on larger independent datasets.

---

## Conclusion

This project successfully developed a complete binary image classification pipeline for distinguishing leaf from non-leaf images.

The workflow included:

    Raw Dataset
         |
         v
    Dataset Inspection
         |
         v
    Cleaning and Filtering
         |
         v
    Duplicate Removal
         |
         v
    Train / Validation / Test Split
         |
         v
    MobileNetV2 Transfer Learning
         |
         v
    Initial Training
         |
         v
    Fine-Tuning
         |
         v
    Independent Evaluation
         |
         v
    Final Model Selection
         |
         v
    Deployment Verification
         |
         v
    Single-Image Prediction

The final fine-tuned MobileNetV2 achieved approximately 99.82% accuracy on the independent 541-image test set.

---

## Project Status

COMPLETED

- Dataset inspected
- Dataset cleaned
- Duplicate images removed
- Dataset split
- Initial model trained
- Initial model evaluated
- Error analysis performed
- Model fine-tuned
- Fine-tuned model evaluated
- Final model selected
- Deployment model created
- Deployment model verified
- Single-image prediction tested
- Project organized
- Documentation created
