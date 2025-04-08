# Part 1: Research & Model Selection

## Overview
The goal of this section is to explore promising models for detecting audio deepfakes, particularly those suitable for real-time or near real-time detection in conversational scenarios.

## 🔍 Model 1: **RawNet3**

###  Key Technical Innovation
- End-to-end deep learning model trained directly on raw audio waveforms.
- Uses residual blocks and gated convolutional layers.
- Embedding extractor followed by classifier.

###  Performance Metrics
- Reported EER (Equal Error Rate): **1.08%** on ASVspoof 2019 LA.
- Competitive with top-performing models in the field.

###  Why It's Promising
- No handcrafted feature extraction required.
- Strong performance and generalizability.
- Learns subtle waveform differences that MFCC might miss.

### ️ Limitations
- Computationally expensive — not feasible for real-time on CPU.
- Requires a large amount of training data and resources.

---

##  Model 2: **ECAPA-TDNN**

###  Key Technical Innovation
- Uses Time-Delay Neural Networks with squeeze-and-excitation blocks and channel attention.
- Extracts robust speaker embeddings (x-vectors).

###  Performance Metrics
- EER: **< 2%** on ASVspoof and other datasets.
- High accuracy in both speaker verification and spoof detection.

###  Why It's Promising
- Compact and efficient.
- Excellent performance in speaker-based classification tasks.
- Could detect inconsistencies in speaker identity for fake audio.

###  Limitations
- Pre-trained models often require fine-tuning.
- Requires more compute than MFCC-based CNNs.

---

##  Model 3: **MFCC + TinyCNN (TinyAudioNet)**

###  Key Technical Innovation
- Simple convolutional model trained on MFCC spectrograms.
- Low computational footprint, ideal for CPU/real-time use.

###  Performance Metrics
- Accuracy achieved in my implementation: **94.6%**.
- Balanced precision and recall for both real and fake classes.

###  Why It's Promising
- Lightweight — feasible for mobile or real-time systems.
- MFCCs are well-established in speech forensics.
- Easily interpretable and easy to train.

###  Limitations
- May miss nuances in raw waveform.
- Dependent on MFCC preprocessing quality.
- Limited ability to adapt to out-of-distribution fakes.

---

##  Summary Table

| Model           | Strengths                          | Weaknesses                        | Suitability         |
|----------------|------------------------------------|-----------------------------------|----------------------|
| RawNet2         | End-to-end, learns raw waveforms  | Heavy compute, not CPU-friendly  | High accuracy, low speed |
| ECAPA-TDNN      | Speaker-focused, robust features  | Needs tuning, GPU preferred      | Great for speaker fraud |
| MFCC + TinyCNN  | Lightweight, real-time capable    | Limited learning capacity        | Best for deployment trials |

---

##  Conclusion
For this assignment, I selected the **MFCC + TinyCNN** model for implementation due to its balance between simplicity, real-time potential, and good classification performance on limited hardware.


# Part 3: Documentation & Analysis

##  Implementation Process

###  Challenges Encountered

- **Data Preparation**:  
  The original dataset had inconsistencies in audio length and format. I addressed this by trimming/padding audio to a fixed length and ensuring consistent sample rates.

- **Limited Compute**:  
  Due to training on CPU, I opted for a lightweight CNN model (TinyAudioNet) on MFCC features instead of heavier architectures like RawNet2 or ECAPA-TDNN.

- **Class Imbalance**:  
  Initially, the dataset had a slight imbalance between real and fake samples. I tackled this using class weights in the loss function and data augmentation (e.g., pitch shift, speed variation) on fake samples.

- **Learning Monitoring**:  
  Added loss/accuracy curves to monitor overfitting/underfitting and model convergence.

###  How Challenges Were Addressed

- Used librosa for consistent MFCC extraction and fixed audio preprocessing pipeline.
- Implemented class-weighted loss using nn.CrossEntropyLoss(weight=...).
- Added learning curves with matplotlib for visual tracking.
- Regular checkpoints and validation accuracy checks during training.

##  Assumptions Made

- Audio clips were assumed to be mono-channel with 16kHz sampling.
- Background noise and artifacts were not explicitly removed, simulating real-world robustness.
- Fake audio primarily originated from TTS systems (as per the dataset), not voice conversion.

##  Analysis Section

###  Why TinyAudioNet + MFCC Was Chosen

- Designed for lightweight inference, making it more suitable for real-time or mobile deployment.
- Compatible with CPU training.
- MFCC features are highly effective in capturing spectral nuances, especially useful for speech forensics.

###  How the Model Works (High-Level)

1. Audio → MFCC features extracted.
2. MFCC → fed into a small CNN with 2 convolutional layers, batch norm, ReLU, pooling.
3. Output → Flattened → Fully connected → Softmax classifier.
4. Loss = CrossEntropyLoss (with class weights).

###  Performance Results

**Best Epoch (20):**
- Train Loss: 0.0504
- Validation Accuracy: 94.62%
- F1-score (Fake): 0.93
- F1-score (Real): 0.94

Classification metrics showed balanced precision/recall and robustness against overfitting.

###  Strengths Observed

- Model converged fast with only 20 epochs.
- Balanced classification across both classes.
- Lightweight and potentially deployable on edge devices.

###  Weaknesses Observed

- Limited fake augmentations (e.g., noise injection not yet implemented).
- Only tested on validation set — no completely unseen test set.
- Doesn't incorporate speaker-specific information or raw waveform learning.

###  Suggestions for Future Improvements

- Implement focal loss to emphasize hard-to-classify samples.
- Add augmentations like noise injection, volume shifts, and reverberation.
- Incorporate separate test set for real-world generalization evaluation.
- Evaluate RawNet2 or ECAPA-TDNN using GPU resources if available.

##  Reflection

**a. What were the most significant challenges in implementing this model?**  
Data preprocessing and real-time augmentation were the most time-consuming.  
Ensuring convergence without overfitting on limited compute was tricky.

**b. How might this approach perform in real-world conditions vs. research datasets?**  
Likely to hold up well in controlled environments.  
Performance may degrade in noisy conditions or with more diverse speaker profiles.  
Lack of speaker variability in training data may cause bias.

**c. What additional data or resources would improve performance?**  
Diverse TTS systems for fake generation (e.g., multilingual, VC systems).  
Real conversational data with spontaneous speech.  
GPU access for training deeper models (RawNet2, ECAPA).

**d. How would you approach deploying this model in a production environment?**  
Convert model to ONNX or TorchScript for inference.  
Deploy on edge (mobile or embedded) with a fast audio preprocessor.  
Add a confidence threshold for downstream decision-making (e.g., flagging uncertain cases).  
Consider a hybrid pipeline with a heavy offline verifier and a lightweight real-time filter.



## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)

### Environment Setup
1. Clone the repository:
```bash
git clone https://github.com/Kss004/DeepFake-Audio-Detection
cd https://github.com/Kss004/DeepFake-Audio-Detection
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

3. Install required packages:
```bash
pip install torch torchaudio librosa matplotlib pandas scikit-learn tqdm numpy seaborn
```

### Dataset Preparation
1. Download the audio datasets (link - https://zenodo.org/records/14498691 (ASVspoof5_protocols.tar.gz, flac_D_aa.tar, flac_t_aa.tar in this case))
2. Place the audio files in the appropriate directories:
   - Real audio samples: `data/train/real/`
   - Fake audio samples: `data/train/fake/`
   - For testing: `data/dev/real/` and `data/dev/fake/`

### Creating the Required Directories and Running the project
1. In order to simplify the whole process, i have done one thing. just run the Audio_Deepfake_detection.ipynb file once. here, doing this will make all the required directories and then also, organise the data into them for your convenience as well. once that is done, you can see the results of the MFCNN model used here.
2. Once that is done, you can run the Main file , ie the Audio_Deepfak_detectino TinyAudioNet.ipynb file 

## Usage

### Training
1. Open `Audio_Deepfake_Detection.ipynb` in Jupyter Notebook:
```bash
jupyter notebook Audio_Deepfake_Detection.ipynb
```
2. Follow the notebook cells to:
   - Load and preprocess the data
   - Train the TinyAudioNet model
   - Evaluate the model performance

### Testing
1. Place your test audio files in the appropriate test directories
2. Use the provided notebooks to run inference on new audio samples

## Model Architecture

### TinyAudioNet
- Input: MFCC features (60 coefficients)
- Architecture:
  - 3 Convolutional layers with batch normalization
  - Global average pooling
  - 2 Fully connected layers
  - Output: Binary classification (real/fake)
