# BrainDiffU-Net: Decentralized Brain MRI Segmentation with U-Net in PyTorch

This repository implements **BrainDiffU-Net**, a decentralized learning framework based on the U-Net architecture for segmenting brain MRI images. It integrates privacy-preserving diffusion techniques to ensure compliance with data privacy regulations, making it suitable for medical image analysis tasks such as brain tumor segmentation.

## Project Structure

The project is divided into five main scripts:

### 1. `dataset.py`
   - **Purpose**: Defines the `BrainMRIDataset` class for loading and preprocessing brain MRI images and their corresponding masks.
   - **Key Functionality**:
     - `__getitem__()`: Loads an MRI image and its corresponding mask, applies preprocessing (resizing and normalization), and converts them to PyTorch tensors.
     - `__len__()`: Returns the total number of samples (image-mask pairs).
       
### 2. `model.py`
   - **Purpose**: Implements the U-Net architecture, a widely used model for segmentation tasks.
   - **Key Functionality**:
     - Encoder-decoder architecture with skip connections for better spatial detail retention.
     - `forward()` method that processes the input through the encoder, bottleneck, and decoder to produce segmentation maps.


### 3. `training.py`
   - **Purpose**: Contains the `train_node_model()` function to train individual models at each decentralized node.
   - **Key Functionality**:
     - Performs training with loss calculation and backpropagation.
     - Tracks metrics such as the Dice coefficient and IoU during each epoch.
     - Implements learning rate scheduling for better convergence.

### 4. `evaluate.py`
   - **Purpose**: Contains the `evaluate_model()` function to visualize the performance of the trained U-Net model on the test dataset.
   - **Key Functionality**:
     - **Visualization**: For a random batch of images from the test set, the original MRI image, the ground truth segmentation mask, and the predicted segmentation mask are displayed side by side. This allows qualitative assessment of the model's performance on unseen data.
     - **Inference**: The model is put into evaluation mode `(model.eval())`, and predictions are generated without gradient computation for efficiency.


### 5. `main.py`
   - **Purpose**: Serves as the main entry point for the project, integrating dataset loading, model training, and model evaluation into one cohesive pipeline.
   - **Key Functionality**:
     - **Dataset Preparation**: Loads the dataset and splits it into training and testing sets. Applies necessary transformations like data augmentation for training.
     - **Model Initialization**: Instantiates the U-Net model, optimizer, and learning rate scheduler.
     - **Training and Validation**: Trains the model using the `train_model()` function and tracks the model’s performance on the validation set.
     - **Evaluation**: After training, the model is evaluated using the `evaluate_model()` function, which visualizes predictions on the test set.

The dataset used in this project comes from two key sources:

1. **Mateusz Buda, Ashirbani Saha, Maciej A. Mazurowski**  
   _"Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm." Computers in Biology and Medicine, 2019._

2. **Maciej A. Mazurowski, Kal Clark, Nicholas M. Czarnek, Parisa Shamsesfandabadi, Katherine B. Peters, Ashirbani Saha**  
   _"Radiogenomics of lower-grade glioma: algorithmically-assessed tumor shape is associated with tumor genomic subtypes and patient outcomes in a multi-institutional study with The Cancer Genome Atlas data." Journal of Neuro-Oncology, 2017._

This dataset contains brain MR images together with manual FLAIR abnormality segmentation masks. The images were obtained from **The Cancer Imaging Archive (TCIA)** and correspond to 110 patients included in **The Cancer Genome Atlas (TCGA)** lower-grade glioma collection. Each patient has at least one fluid-attenuated inversion recovery (FLAIR) sequence and genomic cluster data available.

- Tumor genomic clusters and patient data are provided in the `data.csv` file.
- For more information on genomic data, refer to the publication _"Comprehensive, Integrative Genomic Analysis of Diffuse Lower-Grade Gliomas"_ and supplementary material available [here](https://www.nejm.org/doi/full/10.1056/NEJMoa1402121).




## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/BrainSegNet.git
    cd BrainSegNet
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv brainseg-env
    source brainseg-env/bin/activate  # On Windows use `brainseg-env\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Train the model

Run the `main.py` script to start training the U-Net model:
```bash
python main.py
```

This will:
- Load the brain MRI images and masks.
- Train the U-Net model on the training dataset.
- Save the trained model weights (model.pth) in the working directory.
  
### 2. Evaluate the model
After training, evaluate the model by running the same `main.py` script:
```
python main.py
```

The evaluation part of the script will:
- Load the saved model weights.
- Visualize the original image, the ground truth mask, and the predicted mask for comparison.

## Troubleshooting

If you encounter any issues or errors while running the project, please check the following:

- Ensure all dependencies are installed correctly by running `pip install -r requirements.txt`.
  
- Make sure you are using a compatible version of Python (e.g., Python 3.6 or higher).
 
- Verify that the dataset paths in `main.py` are correct.

If problems persist, feel free to open an issue on GitHub.

## Contributing

Contributions are welcome! If you have suggestions for improvements or bug fixes, please follow these steps:

1. Fork the repository.

2. Create a new branch (`git checkout -b feature-branch`).

3. Make your changes and commit them (`git commit -m 'Add some feature'`).

4. Push to the branch (`git push origin feature-branch`).

5. Open a pull request.

Please ensure your code follows the existing style and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details.

