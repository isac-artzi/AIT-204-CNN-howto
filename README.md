# AIT-204: Build a CNN for Image Classification

An interactive, web-based tutorial that walks students through building a Convolutional Neural Network (CNN) from scratch using PyTorch and the CIFAR-10 dataset.

## What This Tutorial Covers

Across **12 progressive steps**, students build a complete image classifier that achieves 75–80% accuracy on CIFAR-10 (10 classes, 32×32 RGB images). Each step includes theory explanations, hands-on tasks, and copy-ready code blocks.

| Step | Topic |
|------|-------|
| 1 | The Challenge — introducing CIFAR-10 and the classification goal |
| 2 | Project Setup — virtual environments, PyTorch installation |
| 3 | Images as Data — pixels, tensors, normalization |
| 4 | Loading CIFAR-10 — transforms, datasets, DataLoaders |
| 5 | Convolution — filters, kernels, stride, padding |
| 6 | Building Blocks — ReLU, MaxPool, BatchNorm |
| 7 | CNN Architecture — designing and coding the full network |
| 8 | Loss & Optimization — cross-entropy, Adam optimizer |
| 9 | Training Loop — backpropagation, gradient updates |
| 10 | Evaluation — test accuracy, per-class metrics |
| 11 | Improving the Model — augmentation, scheduling, regularization |
| 12 | Complete Application — inference on new images |

## Architecture

The tutorial is a full-stack application with two parts:

- **Backend** — Python / FastAPI API that serves the tutorial content
- **Frontend** — React / TypeScript / Vite app with a three-panel layout (Theory | Tasks | Code)

## Prerequisites

- **Python 3.9+**
- **Node.js 18+**

## Getting Started

### 1. Start the backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

### 2. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

## How to Use

1. Open the app and start at **Step 1**.
2. Read the **Theory** panel (left) to understand the concept.
3. Follow the **Tasks** panel (center) to build each piece of the project.
4. Copy code from the **Code** panel (right) into your own Python project.
5. Use the navigation bar at the top to move between steps.

By the end, students will have built a standalone Python project with this structure:

```
cifar10-cnn/
├── data_utils.py      # Data loading and transforms
├── model.py           # CNN model definition
├── train.py           # Training loop
├── evaluate.py        # Test evaluation and metrics
├── predict.py         # Inference on new images
└── requirements.txt   # PyTorch dependencies
```

## Student Python Dependencies

Students will install these in their own CNN project (not this tutorial app):

- `torch >= 2.0.0`
- `torchvision >= 0.15.0`
- `matplotlib >= 3.7.0`
- `numpy >= 1.24.0`
- `tqdm >= 4.65.0`
