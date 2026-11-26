# AI-TRAINING-2026 Curriculum Models Breakdown

Extracted from the 45-session syllabus at [AI-TRAINING-2026 (`#curriculum-table`)](https://sivamohanbabu.github.io/AI-TRAINING-2026/#curriculum-table).

## Summary Count

- **Specifically named architectures/algorithms:** **16** (`Dense NN`, `CNN`, `YOLO`, `U-Net`, `ViT`, `DETR`, `ResNet50`, `Faster R-CNN`, `Isolation Forest`, `LSTM`, `Q-Learning`, `SARSA`, `DQN`, `PPO`, `SAC`, `Qwen3-VL`)
- **Including general model categories/pipelines (`scikit-learn ML`, `Object Tracking + OCR`, `Transformers/LLMs`, `VLMs`):** **20 total**

*(Sessions 1–7, 15–16, 19–21, 23, 28, 30, 35–37, 41, and Assessments 7, 13, 18, 31, 40, 45 focus on GPU/Kubernetes infrastructure, data engineering, deployment, or evaluations.)*

---

## Breakdown by Category & Session

### 1. Core ML & Neural Networks (3)

| # | Model / Pipeline | Sessions | Details |
| :--- | :--- | :--- | :--- |
| 1 | **scikit-learn ML Pipeline** | Session 9 | Basic supervised ML pipeline (`Input → Model → Prediction`) and evaluation |
| 2 | **Dense Neural Network** | Session 10 | Basic dense classifier implemented in both PyTorch and TensorFlow |
| 3 | **Isolation Forest** | Session 34 | Unsupervised ML anomaly detection using `scikit-learn` |

### 2. Computer Vision, Object Detection & Segmentation (8)

| # | Model / Pipeline | Sessions | Details |
| :--- | :--- | :--- | :--- |
| 4 | **CNN (Convolutional Neural Network)** | Sessions 11–12 | CNN fundamentals, training/inference, and transfer learning |
| 5 | **YOLO** | Sessions 14, 17, 18 | Object detection, custom dataset preparation, and fine-tuning on H200 GPUs |
| 6 | **Object Tracking + OCR** | Session 22 | Combined detection, tracking, and Optical Character Recognition (Smart Lane Violation project) |
| 7 | **U-Net** | Sessions 27, 33 | Medical image / tumour segmentation |
| 8 | **Vision Transformer (ViT)** | Sessions 27, 31 | Tumour segmentation and ViT pipelines |
| 9 | **DETR (DEtection TRansformer)** | Session 32 | Transformer-based object detection |
| 10 | **ResNet50** | Session 32 | Deep residual CNN architecture |
| 11 | **Faster R-CNN** | Session 33 | Two-stage object detection and localization |

### 3. Sequence, NLP & Generative AI / LLMs (2)

| # | Model / Pipeline | Sessions | Details |
| :--- | :--- | :--- | :--- |
| 12 | **LSTM (Long Short-Term Memory)** | Session 32 | Sequential / recurrent neural network architecture |
| 13 | **Transformers & Pretrained LLMs** | Sessions 24–25, 27, 29, 42 | Hugging Face Pipelines & AutoClasses, local vs. API inference, NVIDIA NIM, medical report integration, and NLP tasks (classification, sentiment, QA, summarization, generation) |

### 4. Reinforcement Learning (5)

| # | Algorithm | Sessions | Details |
| :--- | :--- | :--- | :--- |
| 14 | **Q-Learning** | Session 38 | Classic value-based RL algorithm |
| 15 | **SARSA** | Session 38 | On-policy temporal-difference RL algorithm |
| 16 | **Deep Q-Networks (DQN)** | Session 38 | Deep reinforcement learning for value approximation |
| 17 | **Proximal Policy Optimization (PPO)** | Session 39 | Policy-gradient RL algorithm |
| 18 | **Soft Actor-Critic (SAC)** | Session 39 | Entropy-regularized actor-critic RL for continuous control |

### 5. Multimodal & Vision-Language Models (2)

| # | Model / Pipeline | Sessions | Details |
| :--- | :--- | :--- | :--- |
| 19 | **Vision-Language Models (VLMs) / Multimodal Transformers** | Session 43 | Joint vision-and-language processing workflows |
| 20 | **Qwen3-VL** | Session 44 | Domain-specific fine-tuning for an educational chat pipeline |
