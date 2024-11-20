# BrainDiffU-Net: A Distributed Diffusion-Based U-Net for Brain Tumor Segmentation

This repository implements **BrainDiffU-Net**, a decentralized learning framework based on the U-Net architecture for segmenting brain MRI images. It integrates privacy-preserving diffusion techniques to ensure compliance with data privacy regulations, making it suitable for medical image analysis tasks such as brain tumor segmentation.

## Project Structure

The project is divided into five main scripts:

### 1. `data_utils.py`
- **Purpose**: Handles dataset distribution across decentralized nodes.
- **Key Features**:
  - `split_data_into_nodes`: Splits the dataset into subsets for each node to simulate a decentralized environment.
  - Ensures balanced data distribution while redistributing leftover samples to the last node.
       
### 2. `dataset.py`
- **Purpose**: Defines the `BrainMRIDataset` class for loading and preprocessing MRI images and segmentation masks.
- **Key Features**:
  - `__getitem__`: Loads an MRI image and its mask, applies preprocessing (resizing and normalization), and converts them into PyTorch tensors.
  - `__len__`: Returns the total number of samples in the dataset.
  - Supports augmentations using `transforms`.

### 3. `model.py`
- **Purpose**: Implements the U-Net architecture for segmentation tasks.
- **Key Features**:
  - Encoder-decoder structure with skip connections to preserve spatial details.
  - `conv_block` for convolutional operations and `forward()` for the model's forward pass.

### 4. `training.py`
- **Purpose**: Contains the `train_node_model` function to train individual node models in a decentralized setup.
- **Key Features**:
  - Implements the training loop with loss calculation, backpropagation, and optimizer updates.
  - Tracks metrics like Dice Coefficient and IoU during training.
  - Supports learning rate scheduling for improved convergence.

### 5. `metrics.py`
- **Purpose**: Defines metrics for evaluating segmentation quality.
- **Key Features**:
  - `dice_coefficient`: Computes the Dice Similarity Coefficient to measure overlap between predicted and ground truth masks.
  - `iou`: Calculates Intersection over Union for segmentation accuracy.

### 6. `utils.py`
- **Purpose**: Provides utility functions for reproducibility and device initialization.
- **Key Features**:
  - `set_seed`: Ensures reproducibility across different runs by setting seeds for NumPy and PyTorch.
  - `initialize_device`: Detects available GPUs or defaults to CPU for training.

### 7. `main.py`
- **Purpose**: Serves as the main script to integrate all components of the framework.
- **Key Features**:
  - Loads and preprocesses the dataset, splitting it across nodes.
  - Initializes U-Net models, optimizers, schedulers, and DataLoaders.
  - Orchestrates decentralized training with a diffusion process for collaborative learning.
  - Supports multi-GPU training via `DataParallel`.



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
    git clone https://github.com/your-username/BrainDiffU-Net.git
    cd BrainDiffU-Net
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv braindiffunet-env
    source braindiffunet-env/bin/activate  # On Windows use `braindiffunet-env\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Train the model

Run the main.py script to start the decentralized training process:

```
python main.py
```
  
### 2. Evaluate the model
After training, update the evaluation section in `main.py` to visualize predictions:

- Input MRI images
- Ground truth masks
- Predicted masks

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

