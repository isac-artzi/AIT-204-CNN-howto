from app.models import TutorialStep, Task, CodeBlock

STEPS: list[TutorialStep] = [
    # ──────────────────────────────────────────────
    # STEP 1 — The Challenge
    # ──────────────────────────────────────────────
    TutorialStep(
        id=1,
        title="The Challenge",
        subtitle="Image Classification with CIFAR-10",
        theory="""## What Is Image Classification?

Image classification is the task of assigning a **label** to an image from a fixed set of categories. It is one of the foundational problems in computer vision.

### Why Does It Matter?

Image classification powers:
- **Medical imaging** — detecting tumors in X-rays
- **Self-driving cars** — recognizing stop signs, pedestrians
- **Content moderation** — filtering inappropriate images
- **Agriculture** — identifying crop diseases from photos

### The CIFAR-10 Dataset

We will use **CIFAR-10**, a benchmark dataset created by the Canadian Institute For Advanced Research:

| Property | Value |
|----------|-------|
| Images | 60,000 (50k train / 10k test) |
| Size | 32 × 32 pixels |
| Channels | 3 (RGB color) |
| Classes | 10 |

The 10 classes are: **airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck**.

Each image is tiny (32×32), which makes the task challenging — even humans sometimes struggle with these low-resolution images.

### Why Not Just Use Regular Code?

You might think: *"Can't we just write a bunch of if-statements?"*

Traditional programming fails here because:
1. **Pixel values vary wildly** — the same cat can appear in different positions, lighting, and colors
2. **The rules are too complex** to hand-code
3. **We need the computer to learn** the patterns from data

This is where **Convolutional Neural Networks (CNNs)** come in. Instead of writing rules, we show the network thousands of examples and let it learn the patterns automatically.

### What We'll Build

By the end of this tutorial, you will have a working CNN that:
1. Loads and preprocesses the CIFAR-10 dataset
2. Defines a multi-layer convolutional neural network
3. Trains the network using backpropagation
4. Achieves ~75-80% accuracy on the test set
5. Can classify new images into one of 10 categories
""",
        tasks=[
            Task(
                title="Understand the goal",
                instructions="""Read through the theory panel to understand what we're building.

By the end of this tutorial, you will create a project with this structure:

```
cifar10-cnn/
├── data_utils.py      # Data loading and preprocessing
├── model.py           # CNN architecture
├── train.py           # Training loop
├── evaluate.py        # Model evaluation
├── predict.py         # Inference on new images
└── requirements.txt   # Dependencies
```

Each file corresponds to a step in this tutorial. We'll build them one at a time.""",
            ),
            Task(
                title="Prerequisites check",
                instructions="""Make sure you have the following installed:

1. **Python 3.9+** — check with `python --version`
2. **VS Code** — with the Python extension installed
3. **pip** — Python package manager

If you don't have Python, download it from python.org.

Open VS Code and create a new folder called `cifar10-cnn` for our project.""",
            ),
        ],
        code_blocks=[
            CodeBlock(
                filename="terminal",
                language="bash",
                description="Verify your Python installation",
                code="""# Check Python version (need 3.9+)
python --version

# Check pip is available
pip --version""",
            ),
        ],
    ),
    # ──────────────────────────────────────────────
    # STEP 2 — Project Setup
    # ──────────────────────────────────────────────
    TutorialStep(
        id=2,
        title="Project Setup",
        subtitle="Virtual Environment and Dependencies",
        theory="""## Why Virtual Environments?

A **virtual environment** is an isolated Python installation that keeps your project's dependencies separate from your system Python.

### The Problem Without Virtual Environments

Imagine two projects on your computer:
- Project A needs `torch==1.13`
- Project B needs `torch==2.0`

Without virtual environments, installing one version breaks the other. Virtual environments solve this by giving each project its own package directory.

### What Is PyTorch?

**PyTorch** is an open-source deep learning framework developed by Meta AI. We chose it because:

1. **Pythonic** — it feels like regular Python, not a separate language
2. **Dynamic computation graphs** — you can change your network on-the-fly, making debugging easy
3. **Strong community** — most research papers provide PyTorch code
4. **GPU acceleration** — seamlessly moves computations to GPU

### Key PyTorch Packages

| Package | Purpose |
|---------|---------|
| `torch` | Core tensor operations and autograd |
| `torch.nn` | Neural network layers and loss functions |
| `torch.optim` | Optimizers (SGD, Adam, etc.) |
| `torchvision` | Datasets, transforms, and pre-trained models |

### Software Engineering Practice: requirements.txt

Every Python project should have a `requirements.txt` file that lists all dependencies with pinned versions. This ensures anyone can recreate your exact environment.

```
package==version
```

This is a fundamental practice in **reproducible software engineering**. Without it, your code might work on your machine but fail on someone else's.
""",
        tasks=[
            Task(
                title="Create a virtual environment",
                instructions="""Open VS Code's integrated terminal (`Ctrl+\\`` or `Cmd+\\``) and run the commands shown in the code panel to create and activate a virtual environment.

**What's happening:**
- `python -m venv venv` creates a new virtual environment in a `venv/` folder
- `source venv/bin/activate` (Mac/Linux) or `venv\\Scripts\\activate` (Windows) activates it
- Your terminal prompt should now show `(venv)` at the beginning""",
            ),
            Task(
                title="Create requirements.txt",
                instructions="""Create a file called `requirements.txt` in your project root.

Copy the contents from the code panel on the right.

Then install the dependencies with `pip install -r requirements.txt`.

**Note:** This may take a few minutes as PyTorch is a large package (~800MB).""",
            ),
            Task(
                title="Verify installation",
                instructions="""Run the verification script shown in the code panel to make sure everything is installed correctly.

You should see the PyTorch version, whether CUDA (GPU) is available, and a successful tensor operation.""",
            ),
        ],
        code_blocks=[
            CodeBlock(
                filename="terminal",
                language="bash",
                description="Create and activate a virtual environment",
                code="""# Navigate to your project folder
cd cifar10-cnn

# Create a virtual environment
python -m venv venv

# Activate it (Mac/Linux)
source venv/bin/activate

# Activate it (Windows)
# venv\\Scripts\\activate""",
            ),
            CodeBlock(
                filename="requirements.txt",
                language="text",
                description="Project dependencies",
                code="""torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
numpy>=1.24.0
tqdm>=4.65.0""",
            ),
            CodeBlock(
                filename="terminal",
                language="bash",
                description="Install dependencies",
                code="""# Install all dependencies
pip install -r requirements.txt""",
            ),
            CodeBlock(
                filename="verify_setup.py",
                language="python",
                description="Verify that PyTorch is installed correctly",
                code="""import torch
import torchvision

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Quick test: create a tensor and do a simple operation
x = torch.tensor([1.0, 2.0, 3.0])
y = x * 2
print(f"Tensor test: {x} * 2 = {y}")
print("\\nSetup successful! You're ready to build a CNN.")""",
            ),
        ],
    ),
    # ──────────────────────────────────────────────
    # STEP 3 — Images as Data
    # ──────────────────────────────────────────────
    TutorialStep(
        id=3,
        title="Images as Data",
        subtitle="How Computers See Pictures",
        theory="""## From Pixels to Tensors

When you look at a photo of a cat, you see fur, eyes, and ears. A computer sees a **grid of numbers**.

### Pixels and Channels

A digital image is a 2D grid of **pixels**. Each pixel has color values:

- **Grayscale**: 1 value per pixel (0 = black, 255 = white)
- **RGB Color**: 3 values per pixel — Red, Green, Blue (each 0-255)

A 32×32 RGB image is stored as a 3D array of shape `(3, 32, 32)`:
- **3** color channels
- **32** rows of pixels
- **32** columns of pixels

### What Is a Tensor?

A **tensor** is a multi-dimensional array — the fundamental data structure in PyTorch.

| Dimensions | Name | Example |
|-----------|------|---------|
| 0 | Scalar | A single number: `42` |
| 1 | Vector | A list: `[1, 2, 3]` |
| 2 | Matrix | A 2D grid: a grayscale image |
| 3 | 3D Tensor | An RGB image: `(3, 32, 32)` |
| 4 | 4D Tensor | A batch of images: `(N, 3, 32, 32)` |

In PyTorch, a batch of CIFAR-10 images has shape `(N, C, H, W)`:
- **N** = batch size (number of images)
- **C** = channels (3 for RGB)
- **H** = height (32)
- **W** = width (32)

### Normalization

Raw pixel values range from 0 to 255. Neural networks work better with smaller values, so we **normalize** them:

1. **Scale to [0, 1]**: Divide by 255 (`ToTensor()` does this automatically)
2. **Standardize**: Subtract mean and divide by standard deviation

For CIFAR-10, the per-channel statistics are:
- **Mean**: (0.4914, 0.4822, 0.4465)
- **Std**: (0.2470, 0.2435, 0.2616)

After normalization, values are roughly centered around 0 with unit variance. This helps the network train faster and more stably.
""",
        tasks=[
            Task(
                title="Create data_utils.py",
                instructions="""Create a new file called `data_utils.py` in your project root.

This file will handle all data loading and preprocessing. Start by typing the import statements and the constants from the code panel.

**VS Code tip**: You can use `Ctrl+Shift+P` → "Python: Select Interpreter" to make sure VS Code is using your virtual environment.""",
            ),
            Task(
                title="Understand tensor shapes",
                instructions="""Copy and run the exploration script shown in the code panel. This will help you understand:

1. How images are represented as tensors
2. What the shape `(3, 32, 32)` means
3. How pixel values change after normalization

Look at the output carefully — understanding tensor shapes is crucial for debugging CNNs later.""",
            ),
        ],
        code_blocks=[
            CodeBlock(
                filename="data_utils.py",
                language="python",
                description="Data utilities — start with imports and constants",
                code="""import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# CIFAR-10 normalization statistics (pre-computed from training set)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# The 10 classes in CIFAR-10
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)""",
            ),
            CodeBlock(
                filename="explore_tensors.py",
                language="python",
                description="Run this to explore image tensors (educational script)",
                code="""import torch
from torchvision import datasets, transforms

# Load a single image WITHOUT normalization to see raw values
raw_transform = transforms.ToTensor()  # Only scales [0,255] -> [0,1]
dataset = datasets.CIFAR10(root="./data", train=True,
                           download=True, transform=raw_transform)

# Grab the first image and its label
image, label = dataset[0]
classes = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

print(f"Image class: {classes[label]}")
print(f"Tensor type: {type(image)}")
print(f"Tensor shape: {image.shape}")  # torch.Size([3, 32, 32])
print(f"Data type: {image.dtype}")
print(f"Value range: [{image.min():.4f}, {image.max():.4f}]")

# Look at a single pixel (channel, row, col)
print(f"\\nPixel at (0, 16, 16) - Red channel: {image[0, 16, 16]:.4f}")
print(f"Pixel at (1, 16, 16) - Green channel: {image[1, 16, 16]:.4f}")
print(f"Pixel at (2, 16, 16) - Blue channel: {image[2, 16, 16]:.4f}")

# Now with normalization
norm_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])
dataset_norm = datasets.CIFAR10(root="./data", train=True,
                                download=True, transform=norm_transform)
image_norm, _ = dataset_norm[0]

print(f"\\n--- After Normalization ---")
print(f"Value range: [{image_norm.min():.4f}, {image_norm.max():.4f}]")
print(f"Mean per channel: {image_norm.mean(dim=(1,2))}")""",
            ),
        ],
    ),
    # ──────────────────────────────────────────────
    # STEP 4 — Loading CIFAR-10
    # ──────────────────────────────────────────────
    TutorialStep(
        id=4,
        title="Loading CIFAR-10",
        subtitle="Datasets, Transforms, and DataLoaders",
        theory="""## The Data Pipeline

Before a CNN can learn, we need a **data pipeline** that:
1. **Downloads** the raw data
2. **Transforms** each image (resize, normalize, augment)
3. **Batches** images into groups for efficient training
4. **Shuffles** the order each epoch to prevent memorization

### Transforms: Preprocessing Each Image

`torchvision.transforms` provides composable image transformations:

```
transforms.Compose([
    transforms.ToTensor(),        # PIL Image → Tensor, scales to [0,1]
    transforms.Normalize(mean, std), # Standardize values
])
```

Think of `Compose` like a pipeline — each transform is applied in order.

### Dataset: Accessing the Data

`torchvision.datasets.CIFAR10` handles downloading and loading:

```python
dataset = datasets.CIFAR10(
    root="./data",      # Where to store files
    train=True,         # Training set (vs test set)
    download=True,      # Download if not present
    transform=transform # Apply our transforms
)
```

Indexing the dataset returns a `(image_tensor, label)` tuple.

### DataLoader: Batching and Shuffling

Training on one image at a time is inefficient. A `DataLoader` groups images into **batches**:

```python
loader = DataLoader(dataset, batch_size=64, shuffle=True)
```

| Parameter | Purpose |
|-----------|---------|
| `batch_size` | Number of images per batch |
| `shuffle` | Randomize order each epoch |
| `num_workers` | Parallel data loading processes |

**Why batch_size=64?** It's a common default that balances:
- **Memory usage** (larger batches need more GPU memory)
- **Training stability** (larger batches give more stable gradients)
- **Training speed** (larger batches make fewer weight updates per epoch)

### Software Engineering Practice: Functions Over Scripts

Notice how we wrap our data loading in functions rather than writing it as top-level script code. This makes the code:
- **Reusable** — call from training, evaluation, or testing
- **Testable** — easy to write unit tests
- **Configurable** — pass different parameters without editing the file
""",
        tasks=[
            Task(
                title="Add data loading functions to data_utils.py",
                instructions="""Open your `data_utils.py` file and add the `get_transforms()` and `get_data_loaders()` functions from the code panel.

**Key design decisions:**
- We define transforms as a function so we can easily swap them later (e.g., for data augmentation)
- We return both train and test loaders from one function
- We use type hints for clarity and IDE support""",
            ),
            Task(
                title="Test the data pipeline",
                instructions="""Create and run the test script to verify your data pipeline works.

The first run will download CIFAR-10 (~170MB). Subsequent runs will use the cached data.

Verify that:
1. The dataset sizes are correct (50,000 train, 10,000 test)
2. Batch shapes are `(64, 3, 32, 32)` for images
3. Label shapes are `(64,)` — one integer per image""",
            ),
        ],
        code_blocks=[
            CodeBlock(
                filename="data_utils.py",
                language="python",
                description="Complete data_utils.py with loading functions",
                code="""import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# CIFAR-10 normalization statistics (pre-computed from training set)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# The 10 classes in CIFAR-10
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def get_transforms() -> dict[str, transforms.Compose]:
    \"\"\"Return train and test image transforms.\"\"\"
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return {"train": train_transform, "test": test_transform}


def get_data_loaders(
    batch_size: int = 64,
    data_dir: str = "./data",
    num_workers: int = 2,
) -> dict[str, DataLoader]:
    \"\"\"Download CIFAR-10 and return train/test DataLoaders.\"\"\"
    tfms = get_transforms()

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=tfms["train"],
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True,
        transform=tfms["test"],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
    )

    return {"train": train_loader, "test": test_loader}""",
            ),
            CodeBlock(
                filename="test_data.py",
                language="python",
                description="Test script to verify the data pipeline",
                code="""from data_utils import get_data_loaders, CLASSES

# Load data (downloads CIFAR-10 on first run)
loaders = get_data_loaders(batch_size=64)

print(f"Training batches: {len(loaders['train'])}")
print(f"Test batches: {len(loaders['test'])}")

# Inspect one batch
images, labels = next(iter(loaders["train"]))
print(f"\\nBatch image shape: {images.shape}")   # [64, 3, 32, 32]
print(f"Batch label shape: {labels.shape}")       # [64]
print(f"Label values (first 10): {labels[:10]}")
print(f"Classes: {[CLASSES[l] for l in labels[:10].tolist()]}")""",
            ),
        ],
    ),
    # ──────────────────────────────────────────────
    # STEP 5 — Convolution
    # ──────────────────────────────────────────────
    TutorialStep(
        id=5,
        title="Convolution",
        subtitle="The Core Operation Behind CNNs",
        theory="""## What Is Convolution?

**Convolution** is a mathematical operation that slides a small matrix (called a **filter** or **kernel**) across an image, computing element-wise multiplications and sums at each position.

### The Intuition

Imagine sliding a magnifying glass across a photo:
- At each position, the glass examines a small patch
- It computes a single number summarizing what it sees
- The collection of all these numbers forms a **feature map**

### How It Works (Step by Step)

Given a 5×5 input and a 3×3 kernel:

```
Input (5×5):          Kernel (3×3):
1 0 1 0 1             1  0  1
0 1 0 1 0             0  1  0
1 0 1 0 1             1  0  1
0 1 0 1 0
1 0 1 0 1
```

At position (0,0), the kernel overlaps the top-left 3×3 patch:
```
1×1 + 0×0 + 1×1 +
0×0 + 1×1 + 0×0 +
1×1 + 0×0 + 1×1 = 5
```

Slide the kernel one pixel right and repeat. The output is a 3×3 **feature map**.

### Key Parameters

| Parameter | What It Controls |
|-----------|-----------------|
| **Kernel size** | Size of the sliding window (e.g., 3×3) |
| **Stride** | How many pixels to skip when sliding (default: 1) |
| **Padding** | Extra zeros added around the input border |
| **In channels** | Number of input channels (3 for RGB) |
| **Out channels** | Number of filters = number of output feature maps |

### Output Size Formula

```
output_size = (input_size - kernel_size + 2 × padding) / stride + 1
```

Example: Input 32×32, kernel 3×3, padding 1, stride 1:
```
(32 - 3 + 2×1) / 1 + 1 = 32
```
With padding=1, the output keeps the same spatial dimensions.

### Why Convolution Works for Images

1. **Local patterns** — edges, textures are local (a 3×3 filter can detect an edge)
2. **Translation invariance** — the same filter detects a cat eye anywhere in the image
3. **Parameter efficiency** — one 3×3 filter has only 9 weights, shared across all positions (vs millions in a fully connected layer)

### nn.Conv2d in PyTorch

```python
nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
```

This creates 16 filters, each of size 3×3×3 (3 input channels). For a 32×32 input, the output is (16, 32, 32) — 16 feature maps, same spatial size.
""",
        tasks=[
            Task(
                title="Create model.py",
                instructions="""Create a new file called `model.py` in your project root.

We'll build the CNN incrementally over the next few steps. Start with the imports and define a partial model with just the first convolutional layer.

**Think about it:** If our input is a 32×32 RGB image (shape: 3×32×32), and we apply `Conv2d(3, 16, 3, padding=1)`:
- Input: (batch, 3, 32, 32)
- Output: (batch, 16, 32, 32)

The 3 input channels (RGB) become 16 feature maps. Each feature map "looks for" a different pattern.""",
            ),
            Task(
                title="Experiment with convolution",
                instructions="""Run the convolution demo script from the code panel. It shows you:

1. How a convolution layer transforms tensor shapes
2. What the learned kernel weights look like
3. How different kernels produce different feature maps

Watch the shapes carefully — understanding shape transformations is the #1 debugging skill for CNNs.""",
            ),
        ],
        code_blocks=[
            CodeBlock(
                filename="model.py",
                language="python",
                description="Start building the CNN — first convolution layer",
                code="""import torch
import torch.nn as nn


class CIFAR10CNN(nn.Module):
    \"\"\"Convolutional Neural Network for CIFAR-10 classification.

    We build this incrementally — starting with convolution layers.
    \"\"\"

    def __init__(self) -> None:
        super().__init__()

        # First conv layer: 3 input channels (RGB) -> 32 feature maps
        # kernel_size=3, padding=1 keeps spatial dims at 32x32
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32,
            kernel_size=3, padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, 3, 32, 32)
        x = self.conv1(x)
        # Shape: (batch, 32, 32, 32)
        return x""",
            ),
            CodeBlock(
                filename="conv_demo.py",
                language="python",
                description="Interactive demo of convolution operations",
                code="""import torch
import torch.nn as nn

# Create a random "image" batch: 1 image, 3 channels, 8x8
input_tensor = torch.randn(1, 3, 8, 8)
print(f"Input shape: {input_tensor.shape}")

# Apply a conv layer
conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1)
output = conv(input_tensor)
print(f"Output shape: {output.shape}")  # (1, 4, 8, 8)

# Inspect the kernel weights
print(f"\\nKernel shape: {conv.weight.shape}")  # (4, 3, 3, 3)
# 4 filters, each is 3x3x3 (3 channels, 3 height, 3 width)

print(f"Bias shape: {conv.bias.shape}")  # (4,) one bias per filter

# Try different strides and padding
conv_stride2 = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1)
output_stride2 = conv_stride2(input_tensor)
print(f"\\nWith stride=2: {input_tensor.shape} -> {output_stride2.shape}")
# Spatial dims halved: 8x8 -> 4x4

conv_no_pad = nn.Conv2d(3, 4, kernel_size=3, padding=0)
output_no_pad = conv_no_pad(input_tensor)
print(f"With no padding: {input_tensor.shape} -> {output_no_pad.shape}")
# Spatial dims shrink: 8x8 -> 6x6""",
            ),
        ],
    ),
    # ──────────────────────────────────────────────
    # STEP 6 — Building Blocks
    # ──────────────────────────────────────────────
    TutorialStep(
        id=6,
        title="Building Blocks",
        subtitle="ReLU, Pooling, and Batch Normalization",
        theory="""## Activation Functions: ReLU

After convolution, we apply an **activation function** to introduce non-linearity.

Without activation functions, stacking multiple convolution layers would be equivalent to a single layer (linear operations compose into a linear operation). We need non-linearity to learn complex patterns.

### ReLU (Rectified Linear Unit)

```
ReLU(x) = max(0, x)
```

- If x > 0: output is x (pass through)
- If x ≤ 0: output is 0 (block negative values)

**Why ReLU?**
1. **Simple** — just a threshold at zero
2. **Fast** — no expensive operations (unlike sigmoid/tanh)
3. **Effective** — reduces the vanishing gradient problem
4. **Sparse** — zeroes out negative values, creating sparse representations

## Pooling: Reducing Spatial Dimensions

**Pooling** shrinks the feature maps, reducing computation and making the network invariant to small translations.

### Max Pooling

`MaxPool2d(kernel_size=2, stride=2)` takes the maximum value in each 2×2 window:

```
Input (4×4):        Output (2×2):
1  3  2  4          3  4
5  6  1  2    →     8  7
8  2  7  3
4  1  5  6
```

This halves both spatial dimensions: 32×32 → 16×16 → 8×8.

### Why Pool?

1. **Reduces computation** — fewer pixels to process in later layers
2. **Prevents overfitting** — fewer parameters
3. **Translation invariance** — a feature detected at position (10,10) or (11,11) produces the same pooled result

## Batch Normalization

**BatchNorm** normalizes the outputs of a layer across the batch dimension:

```
y = (x - mean) / sqrt(var + ε) × γ + β
```

Where γ (scale) and β (shift) are learnable parameters.

### Why Batch Normalization?

1. **Faster training** — allows higher learning rates
2. **Regularization** — slight noise from batch statistics acts as regularizer
3. **Stable gradients** — prevents internal covariate shift

### The Conv Block Pattern

In practice, these three operations are always used together:

```
Conv2d → BatchNorm2d → ReLU → MaxPool2d
```

This is the fundamental building block of most CNNs. We'll use this pattern repeatedly.
""",
        tasks=[
            Task(
                title="Understand each building block",
                instructions="""Before adding code, make sure you understand each component:

1. **ReLU**: Takes a tensor, replaces all negative values with 0
2. **MaxPool2d**: Takes a tensor, shrinks spatial dims by 2× (takes max in each 2×2 window)
3. **BatchNorm2d**: Normalizes each feature map across the batch

Run the building blocks demo from the code panel to see how each operation changes the tensor shapes and values.""",
            ),
            Task(
                title="Update model.py with building blocks",
                instructions="""Don't modify `model.py` yet — we'll do the full architecture in the next step.

For now, understand that our CNN will use this repeated pattern:

```
Conv2d(in, out, 3, padding=1)  →  keeps spatial size
BatchNorm2d(out)                →  normalizes
ReLU()                          →  non-linearity
MaxPool2d(2, 2)                 →  halves spatial size
```

Shape progression for 32×32 input:
- After conv block 1: (32, 32, 32) → pool → (32, 16, 16)
- After conv block 2: (64, 16, 16) → pool → (64, 8, 8)
- After conv block 3: (128, 8, 8) → pool → (128, 4, 4)""",
            ),
        ],
        code_blocks=[
            CodeBlock(
                filename="blocks_demo.py",
                language="python",
                description="Demo of ReLU, pooling, and batch normalization",
                code="""import torch
import torch.nn as nn

# ---- ReLU Demo ----
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
relu = nn.ReLU()
print(f"ReLU input:  {x}")
print(f"ReLU output: {relu(x)}")
# [-2, -1, 0, 1, 2] -> [0, 0, 0, 1, 2]

# ---- MaxPool2d Demo ----
# Create a 1x1x4x4 tensor (batch=1, channels=1, 4x4 spatial)
feature_map = torch.tensor([[
    [1.0, 3.0, 2.0, 4.0],
    [5.0, 6.0, 1.0, 2.0],
    [8.0, 2.0, 7.0, 3.0],
    [4.0, 1.0, 5.0, 6.0],
]]).unsqueeze(0)

pool = nn.MaxPool2d(kernel_size=2, stride=2)
pooled = pool(feature_map)
print(f"\\nMaxPool input shape:  {feature_map.shape}")  # (1,1,4,4)
print(f"MaxPool output shape: {pooled.shape}")           # (1,1,2,2)
print(f"MaxPool output:\\n{pooled.squeeze()}")

# ---- BatchNorm Demo ----
# 4 images, 3 channels, 2x2 spatial
batch = torch.randn(4, 3, 2, 2) * 10 + 5  # Mean ~5, std ~10
bn = nn.BatchNorm2d(3)
normalized = bn(batch)
print(f"\\nBefore BatchNorm - mean: {batch.mean():.2f}, std: {batch.std():.2f}")
print(f"After BatchNorm  - mean: {normalized.mean():.2f}, std: {normalized.std():.2f}")

# ---- Full Conv Block ----
print("\\n--- Full Conv Block ---")
block = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
)
images = torch.randn(8, 3, 32, 32)
output = block(images)
print(f"Input:  {images.shape}")   # (8, 3, 32, 32)
print(f"Output: {output.shape}")   # (8, 32, 16, 16)""",
            ),
        ],
    ),
    # ──────────────────────────────────────────────
    # STEP 7 — CNN Architecture
    # ──────────────────────────────────────────────
    TutorialStep(
        id=7,
        title="CNN Architecture",
        subtitle="Designing the Complete Network",
        theory="""## Designing a CNN Architecture

Now we put together all the building blocks into a complete network. A CNN typically has two parts:

### 1. Feature Extractor (Convolutional Layers)

The convolutional layers progressively extract features:
- **Early layers** detect simple features: edges, corners, colors
- **Middle layers** detect textures: fur patterns, wheel shapes
- **Deep layers** detect high-level features: faces, wheels, wings

Each conv block:
1. Increases the number of channels (more features)
2. Decreases the spatial dimensions (pooling)

Our architecture:
```
Input: (3, 32, 32)

Conv Block 1: Conv(3→32) + BN + ReLU + Pool
  → (32, 16, 16) — 32 simple feature maps

Conv Block 2: Conv(32→64) + BN + ReLU + Pool
  → (64, 8, 8) — 64 more complex features

Conv Block 3: Conv(64→128) + BN + ReLU + Pool
  → (128, 4, 4) — 128 high-level features
```

### 2. Classifier (Fully Connected Layers)

After the conv layers, we **flatten** the 3D feature tensor into a 1D vector and pass it through fully connected (linear) layers:

```
Flatten: (128, 4, 4) → (2048,)
FC1: 2048 → 256 + ReLU + Dropout
FC2: 256 → 10  (10 classes)
```

### The nn.Module Pattern

Every PyTorch network inherits from `nn.Module` and implements two methods:

1. **`__init__`** — define all layers
2. **`forward`** — define how data flows through the layers

```python
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 5)

    def forward(self, x):
        return self.layer(x)
```

PyTorch tracks all layers defined in `__init__` automatically — this is how it knows which parameters to train.

### Dropout: A Regularization Technique

**Dropout** randomly sets a fraction of neurons to zero during training:

```python
nn.Dropout(p=0.5)  # 50% of neurons zeroed each forward pass
```

This prevents **co-adaptation** — neurons can't rely on specific other neurons, forcing the network to learn more robust features. During evaluation, dropout is turned off.

### Total Parameters

Our network has:
- Conv layers: ~110K parameters
- FC layers: ~530K parameters
- Total: ~640K parameters

This is small by modern standards but effective for CIFAR-10.
""",
        tasks=[
            Task(
                title="Implement the full CNN",
                instructions="""Replace the contents of `model.py` with the complete CNN architecture from the code panel.

Take time to trace through the shapes:
1. Input: (batch, 3, 32, 32)
2. After block 1: (batch, 32, 16, 16)
3. After block 2: (batch, 64, 8, 8)
4. After block 3: (batch, 128, 4, 4)
5. After flatten: (batch, 2048)
6. After fc1: (batch, 256)
7. Output: (batch, 10)

**VS Code tip**: Hover over `nn.Conv2d` to see its documentation.""",
            ),
            Task(
                title="Test the model",
                instructions="""Run the model test script from the code panel. Verify:

1. The model accepts a batch of 32×32 RGB images
2. The output has 10 values per image (one per class)
3. The total parameter count is reasonable (~640K)

If you get a shape error, the most common cause is a mismatch in the flatten size. The formula is: `out_channels × (32 / (2^num_pools))²`""",
            ),
        ],
        code_blocks=[
            CodeBlock(
                filename="model.py",
                language="python",
                description="Complete CNN architecture",
                code="""import torch
import torch.nn as nn


class CIFAR10CNN(nn.Module):
    \"\"\"Convolutional Neural Network for CIFAR-10 classification.

    Architecture:
        3 conv blocks (Conv → BN → ReLU → MaxPool) followed by
        2 fully connected layers with dropout.
    \"\"\"

    def __init__(self) -> None:
        super().__init__()

        # --- Feature extractor (convolutional layers) ---

        # Block 1: (3, 32, 32) → (32, 16, 16)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: (32, 16, 16) → (64, 8, 8)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: (64, 8, 8) → (128, 4, 4)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Classifier (fully connected layers) ---

        # 128 channels × 4 × 4 spatial = 2048 features
        self.classifier = nn.Sequential(
            nn.Flatten(),                # (128, 4, 4) → (2048,)
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),          # 10 CIFAR-10 classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, 32, 32).

        Returns:
            Logits of shape (batch, 10).
        \"\"\"
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x""",
            ),
            CodeBlock(
                filename="test_model.py",
                language="python",
                description="Test the CNN model",
                code="""import torch
from model import CIFAR10CNN

# Create the model
model = CIFAR10CNN()

# Create a fake batch: 4 images, 3 channels, 32x32
fake_batch = torch.randn(4, 3, 32, 32)

# Forward pass
output = model(fake_batch)
print(f"Input shape:  {fake_batch.shape}")  # (4, 3, 32, 32)
print(f"Output shape: {output.shape}")      # (4, 10)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable:,}")

# Print model architecture
print(f"\\nModel architecture:\\n{model}")""",
            ),
        ],
    ),
    # ──────────────────────────────────────────────
    # STEP 8 — Loss & Optimization
    # ──────────────────────────────────────────────
    TutorialStep(
        id=8,
        title="Loss & Optimization",
        subtitle="Teaching the Network to Learn",
        theory="""## How Does a Network Learn?

Training a neural network is an optimization problem:
1. **Forward pass** — feed an image through the network, get a prediction
2. **Compute loss** — measure how wrong the prediction is
3. **Backward pass** — compute gradients (how to adjust each weight)
4. **Update weights** — nudge weights in the direction that reduces loss

Repeat thousands of times until the loss is small.

### Cross-Entropy Loss

For classification, we use **Cross-Entropy Loss**:

```
Loss = -log(probability of correct class)
```

Our network outputs **logits** — raw scores for each class. Cross-entropy:
1. Converts logits to probabilities using **softmax**: `p_i = e^{z_i} / Σe^{z_j}`
2. Takes the negative log of the correct class's probability

| Scenario | Correct class probability | Loss |
|----------|--------------------------|------|
| Perfect | 1.0 | 0.0 |
| Good | 0.8 | 0.22 |
| Uncertain | 0.5 | 0.69 |
| Wrong | 0.1 | 2.30 |

The loss is high when the network is wrong and low when it's right.

### Optimizers

An **optimizer** updates the network weights based on the computed gradients.

#### SGD (Stochastic Gradient Descent)

```
weight = weight - learning_rate × gradient
```

Simple but effective. The **learning rate** controls how big each step is:
- Too large → overshoots, training diverges
- Too small → takes forever to converge

#### Adam (Adaptive Moment Estimation)

Adam is a more sophisticated optimizer that:
1. Keeps a running average of gradients (**momentum**)
2. Adapts the learning rate for each parameter separately
3. Generally converges faster than SGD

We'll use Adam with a learning rate of 0.001 — a robust default.

### Learning Rate: The Most Important Hyperparameter

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

| Learning Rate | Typical Behavior |
|--------------|-----------------|
| 0.1 | Too high — loss oscillates or diverges |
| 0.01 | May work for SGD |
| 0.001 | Good default for Adam |
| 0.0001 | Slow but stable |

### Software Engineering Practice: Configuration

Notice how we pass `learning_rate` and `epochs` as function parameters instead of hard-coding them. This makes the code configurable without editing source files.
""",
        tasks=[
            Task(
                title="Create train.py",
                instructions="""Create a new file called `train.py`. Start with the imports and the training setup function from the code panel.

Notice how we:
1. Check for GPU availability and set the device
2. Move both the model and data to the same device
3. Use `model.train()` to enable dropout and batch norm training behavior

**Key concept**: `model.parameters()` returns all learnable weights. The optimizer needs this list to know what to update.""",
            ),
            Task(
                title="Test loss computation",
                instructions="""Run the loss demo script to understand how cross-entropy loss works.

Key observations:
1. A confident correct prediction has low loss
2. A wrong prediction has high loss
3. `torch.argmax(logits)` gives the predicted class""",
            ),
        ],
        code_blocks=[
            CodeBlock(
                filename="train.py",
                language="python",
                description="Training setup — imports, device selection, loss, and optimizer",
                code="""import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_utils import get_data_loaders
from model import CIFAR10CNN


def get_device() -> torch.device:
    \"\"\"Return the best available device (GPU if available).\"\"\"
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    return torch.device("cpu")""",
            ),
            CodeBlock(
                filename="loss_demo.py",
                language="python",
                description="Understand cross-entropy loss",
                code="""import torch
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()

# Scenario 1: Confident and CORRECT
# Model strongly predicts class 2 (logit=5.0), true label is 2
logits = torch.tensor([[0.1, 0.2, 5.0, 0.1]])
label = torch.tensor([2])
loss = loss_fn(logits, label)
print(f"Confident & correct:   loss = {loss.item():.4f}")

# Scenario 2: Confident but WRONG
# Model strongly predicts class 0 (logit=5.0), true label is 2
logits = torch.tensor([[5.0, 0.2, 0.1, 0.1]])
label = torch.tensor([2])
loss = loss_fn(logits, label)
print(f"Confident & wrong:     loss = {loss.item():.4f}")

# Scenario 3: Uncertain
# Model gives roughly equal scores to all classes
logits = torch.tensor([[1.0, 1.0, 1.1, 1.0]])
label = torch.tensor([2])
loss = loss_fn(logits, label)
print(f"Uncertain:             loss = {loss.item():.4f}")

# Show softmax probabilities
probs = torch.softmax(logits, dim=1)
print(f"\\nSoftmax probabilities: {probs.squeeze()}")
print(f"Predicted class: {torch.argmax(probs).item()}")""",
            ),
        ],
    ),
    # ──────────────────────────────────────────────
    # STEP 9 — Training Loop
    # ──────────────────────────────────────────────
    TutorialStep(
        id=9,
        title="Training Loop",
        subtitle="Teaching the Network with Backpropagation",
        theory="""## The Training Loop

The training loop is the heart of deep learning. For each **epoch** (one pass through the full dataset):

```
for each epoch:
    for each batch of images:
        1. Forward pass: predictions = model(images)
        2. Compute loss: loss = loss_fn(predictions, labels)
        3. Zero gradients: optimizer.zero_grad()
        4. Backward pass: loss.backward()
        5. Update weights: optimizer.step()
```

### Why Zero Gradients?

PyTorch **accumulates** gradients by default. If you don't zero them, gradients from the current batch are added to gradients from the previous batch. `optimizer.zero_grad()` resets them to zero before each new computation.

### What Does `loss.backward()` Do?

This triggers **backpropagation** — the algorithm that computes how much each weight contributed to the loss:

1. Starting from the loss, trace backward through every operation
2. At each layer, compute the gradient (partial derivative) using the chain rule
3. Store the gradient in each parameter's `.grad` attribute

### What Does `optimizer.step()` Do?

After gradients are computed, `optimizer.step()` updates each weight:

```
weight.data -= learning_rate * weight.grad
```

(Adam uses a more sophisticated update rule, but the idea is the same.)

### Epochs and Batches

| Term | Meaning |
|------|---------|
| **Batch** | A group of images processed together (e.g., 64) |
| **Iteration** | One weight update (one batch) |
| **Epoch** | One full pass through the entire training set |

For CIFAR-10 with batch_size=64:
- 50,000 images / 64 per batch = **782 iterations per epoch**
- Training for 20 epochs = 15,640 total weight updates

### Tracking Progress

We track two metrics during training:
1. **Loss** — should decrease over time
2. **Accuracy** — should increase over time

A common pattern:
- **Epoch 1**: ~40% accuracy (random would be 10%)
- **Epoch 5**: ~60% accuracy
- **Epoch 10**: ~70% accuracy
- **Epoch 20**: ~75-80% accuracy

### Software Engineering Practice: Progress Bars

We use `tqdm` for progress bars — a small but important UX improvement that makes long training runs less anxiety-inducing. Always give your users feedback about what's happening.
""",
        tasks=[
            Task(
                title="Implement the training loop",
                instructions="""Add the `train_one_epoch` and `train` functions to your `train.py` file.

The code in the panel shows the complete `train.py`. Replace your current file with this code.

**Key things to notice:**
1. `model.train()` enables training-mode behaviors (dropout, batch norm)
2. We zero gradients BEFORE the backward pass
3. We accumulate running loss and accuracy for reporting
4. `tqdm` shows a progress bar for each epoch""",
            ),
            Task(
                title="Run training",
                instructions="""Run the training script. With the default settings (20 epochs), expect:

- **CPU**: 15-30 minutes
- **GPU (CUDA/MPS)**: 3-5 minutes

Watch the loss decrease and accuracy increase each epoch. If loss isn't decreasing, something is wrong.

**Tip**: Start with 5 epochs first to make sure everything works, then run the full 20.

```
python train.py
```""",
            ),
        ],
        code_blocks=[
            CodeBlock(
                filename="train.py",
                language="python",
                description="Complete training script",
                code="""import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_utils import get_data_loaders
from model import CIFAR10CNN


def get_device() -> torch.device:
    \"\"\"Return the best available device (GPU if available).\"\"\"
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    \"\"\"Train for one epoch. Returns loss and accuracy.\"\"\"
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return {"loss": epoch_loss, "accuracy": epoch_acc}


def train(
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.001,
) -> nn.Module:
    \"\"\"Full training pipeline. Returns the trained model.\"\"\"
    device = get_device()
    print(f"Using device: {device}")

    # Data
    loaders = get_data_loaders(batch_size=batch_size)

    # Model
    model = CIFAR10CNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1, epochs + 1):
        metrics = train_one_epoch(model, loaders["train"],
                                  loss_fn, optimizer, device)
        print(
            f"Epoch {epoch:2d}/{epochs} — "
            f"Loss: {metrics['loss']:.4f}, "
            f"Accuracy: {metrics['accuracy']:.2f}%"
        )

    # Save the trained model
    torch.save(model.state_dict(), "cifar10_cnn.pth")
    print("\\nModel saved to cifar10_cnn.pth")
    return model


if __name__ == "__main__":
    train()""",
            ),
        ],
    ),
    # ──────────────────────────────────────────────
    # STEP 10 — Evaluation
    # ──────────────────────────────────────────────
    TutorialStep(
        id=10,
        title="Evaluation",
        subtitle="Measuring Model Performance",
        theory="""## Why Evaluate on Test Data?

During training, the model sees the **training set** repeatedly. We need to check its performance on data it has **never seen** — the **test set**.

If the model performs:
- **Well on train, well on test** → Good! The model generalizes.
- **Well on train, poorly on test** → Overfitting! The model memorized training data.
- **Poorly on both** → Underfitting. The model is too simple or needs more training.

### model.eval() vs model.train()

```python
model.eval()   # Evaluation mode
model.train()  # Training mode
```

These switch important behaviors:
| Behavior | train() | eval() |
|----------|---------|--------|
| Dropout | Randomly zeros neurons | Disabled (uses all neurons) |
| BatchNorm | Uses batch statistics | Uses running averages |

### torch.no_grad()

During evaluation, we don't need gradients (we're not updating weights). Wrapping code in `torch.no_grad()` disables gradient computation, which:
1. **Saves memory** — no need to store intermediate values for backprop
2. **Speeds up** computation

```python
with torch.no_grad():
    outputs = model(images)
```

### Evaluation Metrics

#### Overall Accuracy
```
accuracy = correct_predictions / total_predictions
```

Simple but can be misleading if classes are imbalanced.

#### Per-Class Accuracy

Shows which classes the model handles well and which it struggles with. For CIFAR-10, you might find:
- **Easy**: truck, ship, airplane (distinct shapes)
- **Hard**: cat vs dog, deer vs horse (similar features)

#### Confusion Matrix

A table showing actual vs. predicted classes. Helps identify specific misclassification patterns.

### Software Engineering Practice: Separation of Concerns

We put evaluation in its own file (`evaluate.py`) separate from training (`train.py`). This allows:
- Evaluating a pre-trained model without retraining
- Running evaluation in CI/CD pipelines
- Swapping different evaluation metrics independently
""",
        tasks=[
            Task(
                title="Create evaluate.py",
                instructions="""Create a new file called `evaluate.py` with the code from the code panel.

This script:
1. Loads the saved model weights
2. Runs inference on the entire test set (no gradients)
3. Computes overall accuracy and per-class accuracy
4. Displays results in a formatted table""",
            ),
            Task(
                title="Run evaluation",
                instructions="""After training completes (Step 9), run:

```
python evaluate.py
```

Expected results:
- Overall accuracy: ~75-80%
- Some classes (ship, truck) will score higher
- Confusing pairs (cat/dog, deer/horse) will score lower

**If accuracy is below 60%**, something went wrong in training. Check that:
1. Your model architecture matches Step 7
2. You trained for enough epochs (at least 15-20)
3. The data is being normalized correctly""",
            ),
        ],
        code_blocks=[
            CodeBlock(
                filename="evaluate.py",
                language="python",
                description="Model evaluation script",
                code="""import torch
import torch.nn as nn

from data_utils import get_data_loaders, CLASSES
from model import CIFAR10CNN
from train import get_device


def evaluate(model_path: str = "cifar10_cnn.pth") -> dict:
    \"\"\"Evaluate a trained model on the CIFAR-10 test set.\"\"\"
    device = get_device()

    # Load model
    model = CIFAR10CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load test data
    loaders = get_data_loaders()
    test_loader = loaders["test"]

    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    # Print results
    overall_acc = 100.0 * correct / total
    print(f"Overall Test Accuracy: {overall_acc:.2f}%")
    print(f"({'='*40})")

    print(f"\\n{'Class':<14} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"{'-'*42}")
    for i in range(10):
        acc = 100.0 * class_correct[i] / class_total[i]
        print(f"{CLASSES[i]:<14} {class_correct[i]:>8} "
              f"{class_total[i]:>8} {acc:>9.2f}%")

    return {
        "overall_accuracy": overall_acc,
        "class_accuracy": {
            CLASSES[i]: 100.0 * class_correct[i] / class_total[i]
            for i in range(10)
        },
    }


if __name__ == "__main__":
    evaluate()""",
            ),
        ],
    ),
    # ──────────────────────────────────────────────
    # STEP 11 — Improving the Model
    # ──────────────────────────────────────────────
    TutorialStep(
        id=11,
        title="Improving the Model",
        subtitle="Data Augmentation and Regularization",
        theory="""## Pushing Past 80%: Techniques for Improvement

Our baseline model achieves ~75-80%. Let's explore techniques to improve it.

### 1. Data Augmentation

**Data augmentation** creates new training examples by applying random transformations to existing images:

```
Original image → Random flip → Random crop → Color jitter → Augmented image
```

This is powerful because:
- **More data** without collecting new images
- **Regularization** — the model never sees the exact same image twice
- **Invariance** — teaches the model that a flipped cat is still a cat

Common augmentations for CIFAR-10:

| Transform | What It Does |
|-----------|-------------|
| `RandomHorizontalFlip` | 50% chance to mirror left-right |
| `RandomCrop` with padding | Shift the image randomly |
| `ColorJitter` | Random brightness, contrast, saturation |
| `RandomRotation` | Rotate by a small angle |

**Important:** Only augment the training set, never the test set!

### 2. Learning Rate Scheduling

Instead of using a fixed learning rate, we can decrease it during training:

- **Start high** (0.001) — make big steps to find the right region
- **Decrease gradually** — make smaller steps to fine-tune

`StepLR(optimizer, step_size=7, gamma=0.1)` multiplies the learning rate by 0.1 every 7 epochs:
- Epochs 1-7: lr = 0.001
- Epochs 8-14: lr = 0.0001
- Epochs 15-20: lr = 0.00001

### 3. More Dropout

Adding dropout to the convolutional layers (not just the classifier) provides additional regularization:

```python
nn.Dropout2d(p=0.1)  # Zeros out entire feature maps
```

Note: `Dropout2d` drops entire channels, while `Dropout` drops individual neurons. For conv layers, `Dropout2d` is more appropriate.

### 4. Weight Decay (L2 Regularization)

Weight decay penalizes large weights, preventing the model from being overly complex:

```python
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

### Combining Improvements

Each technique alone gives a small boost. Combined, they can push accuracy from ~78% to ~85%+:

| Technique | Estimated Improvement |
|-----------|---------------------|
| Data augmentation | +3-5% |
| LR scheduling | +1-2% |
| Weight decay | +1-2% |
| More epochs | +1-2% |
""",
        tasks=[
            Task(
                title="Update data_utils.py with augmentation",
                instructions="""Modify the `get_transforms()` function in `data_utils.py` to add data augmentation to the training transforms.

Replace only the `get_transforms` function with the improved version from the code panel. Notice:
- Training transforms now include `RandomHorizontalFlip` and `RandomCrop`
- Test transforms remain unchanged (we always evaluate on clean data)""",
            ),
            Task(
                title="Update training with LR scheduling",
                instructions="""Update your `train.py` with the improved version from the code panel. Key changes:
1. Added `weight_decay` to the optimizer
2. Added a `StepLR` learning rate scheduler
3. Added `scheduler.step()` after each epoch

Run training again with the improvements:
```
python train.py
```

Compare the final accuracy with your baseline model.""",
            ),
        ],
        code_blocks=[
            CodeBlock(
                filename="data_utils.py",
                language="python",
                description="Updated get_transforms with data augmentation",
                code="""import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# CIFAR-10 normalization statistics (pre-computed from training set)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# The 10 classes in CIFAR-10
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def get_transforms() -> dict[str, transforms.Compose]:
    \"\"\"Return train and test image transforms.

    Training transforms include data augmentation for better generalization.
    Test transforms only normalize (no augmentation on test data!).
    \"\"\"
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return {"train": train_transform, "test": test_transform}


def get_data_loaders(
    batch_size: int = 64,
    data_dir: str = "./data",
    num_workers: int = 2,
) -> dict[str, DataLoader]:
    \"\"\"Download CIFAR-10 and return train/test DataLoaders.\"\"\"
    tfms = get_transforms()

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=tfms["train"],
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True,
        transform=tfms["test"],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
    )

    return {"train": train_loader, "test": test_loader}""",
            ),
            CodeBlock(
                filename="train.py",
                language="python",
                description="Improved training with LR scheduling and weight decay",
                code="""import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from data_utils import get_data_loaders
from model import CIFAR10CNN


def get_device() -> torch.device:
    \"\"\"Return the best available device (GPU if available).\"\"\"
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    \"\"\"Train for one epoch. Returns loss and accuracy.\"\"\"
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return {"loss": epoch_loss, "accuracy": epoch_acc}


def train(
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.001,
) -> nn.Module:
    \"\"\"Full training pipeline. Returns the trained model.\"\"\"
    device = get_device()
    print(f"Using device: {device}")

    # Data
    loaders = get_data_loaders(batch_size=batch_size)

    # Model
    model = CIFAR10CNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss and optimizer (with weight decay for regularization)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=1e-4)

    # Learning rate scheduler: reduce LR by 10x every 7 epochs
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # Training loop
    for epoch in range(1, epochs + 1):
        metrics = train_one_epoch(model, loaders["train"],
                                  loss_fn, optimizer, device)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:2d}/{epochs} — "
            f"Loss: {metrics['loss']:.4f}, "
            f"Accuracy: {metrics['accuracy']:.2f}%, "
            f"LR: {current_lr:.6f}"
        )

    # Save the trained model
    torch.save(model.state_dict(), "cifar10_cnn.pth")
    print("\\nModel saved to cifar10_cnn.pth")
    return model


if __name__ == "__main__":
    train()""",
            ),
        ],
    ),
    # ──────────────────────────────────────────────
    # STEP 12 — Complete Application
    # ──────────────────────────────────────────────
    TutorialStep(
        id=12,
        title="Complete Application",
        subtitle="Putting It All Together",
        theory="""## From Model to Application

You've built a complete CNN pipeline! Let's review what we created and add the final piece: **inference on new images**.

### Project Architecture Review

```
cifar10-cnn/
├── data_utils.py      # Data pipeline (transforms, DataLoader)
├── model.py           # CNN architecture (CIFAR10CNN)
├── train.py           # Training loop with LR scheduling
├── evaluate.py        # Test set evaluation
├── predict.py         # Inference on new images (this step!)
└── requirements.txt   # Dependencies
```

Each file has a **single responsibility**:
- `data_utils.py` — knows how to load and transform data
- `model.py` — defines the network structure
- `train.py` — handles the training process
- `evaluate.py` — measures performance
- `predict.py` — uses the trained model on new inputs

### Inference Pipeline

To classify a new image:
1. **Load** the image
2. **Transform** it (resize, normalize — same transforms as test data)
3. **Pass** it through the model
4. **Interpret** the output (softmax → probabilities)

### What You've Learned

| Topic | Key Takeaway |
|-------|-------------|
| **CNNs** | Convolutional layers extract spatial features from images |
| **Convolution** | A sliding filter that detects local patterns |
| **Pooling** | Reduces spatial dimensions while keeping important features |
| **Batch Norm** | Stabilizes and accelerates training |
| **Cross-Entropy** | Loss function for classification tasks |
| **Adam** | Adaptive optimizer that works well as a default |
| **Data Augmentation** | Creates training variety for better generalization |
| **LR Scheduling** | Reduces learning rate over time for fine-tuning |
| **train/eval modes** | Different behavior for dropout and batch norm |

### Software Engineering Practices

Throughout this project, you've followed these professional practices:
1. **Virtual environments** — isolated dependencies
2. **requirements.txt** — reproducible installations
3. **Modular design** — separate files for separate concerns
4. **Functions over scripts** — reusable, testable code
5. **Type hints** — clear interfaces
6. **Docstrings** — documented functions
7. **Configuration via parameters** — not hard-coded values

### Next Steps

Now that you have a working CNN, here are ideas to explore:
- **Try different architectures**: Add more layers, change kernel sizes
- **Use pre-trained models**: `torchvision.models.resnet18(pretrained=True)`
- **Try a different dataset**: Fashion-MNIST, STL-10, or your own images
- **Visualize filters**: See what patterns each conv layer detects
- **Deploy the model**: Create a web API using FastAPI
""",
        tasks=[
            Task(
                title="Create predict.py",
                instructions="""Create the final file: `predict.py`. This script:

1. Loads the trained model
2. Takes an image (from CIFAR-10 test set as demo)
3. Runs inference and shows top-3 predictions with confidence scores

Copy the code from the code panel.""",
            ),
            Task(
                title="Run the complete pipeline",
                instructions="""Run through the entire pipeline from start to finish:

```bash
# 1. Train the model (if not already done)
python train.py

# 2. Evaluate on test set
python evaluate.py

# 3. Run predictions on sample images
python predict.py
```

Congratulations! You've built a complete CNN image classifier from scratch.

**Final project structure:**
```
cifar10-cnn/
├── data_utils.py
├── model.py
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
├── cifar10_cnn.pth        (generated by training)
└── data/                  (downloaded dataset)
    └── cifar-10-batches-py/
```""",
            ),
        ],
        code_blocks=[
            CodeBlock(
                filename="predict.py",
                language="python",
                description="Inference script — classify images with the trained model",
                code="""import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from data_utils import CIFAR10_MEAN, CIFAR10_STD, CLASSES
from model import CIFAR10CNN
from train import get_device


def predict(model_path: str = "cifar10_cnn.pth") -> None:
    \"\"\"Load a trained model and predict on sample test images.\"\"\"
    device = get_device()

    # Load model
    model = CIFAR10CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load some test images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_dataset = datasets.CIFAR10(
        root="./data", train=False,
        download=True, transform=transform,
    )

    # Predict on 10 random samples
    print("Predictions on test images:")
    print(f"{'True Label':<14} {'Predicted':<14} {'Confidence':>10}  Top-3")
    print(f"{'-'*60}")

    indices = torch.randperm(len(test_dataset))[:10]
    with torch.no_grad():
        for idx in indices:
            image, true_label = test_dataset[idx.item()]
            image = image.unsqueeze(0).to(device)  # Add batch dim

            # Forward pass
            logits = model(image)
            probs = F.softmax(logits, dim=1).squeeze()

            # Top-3 predictions
            top3_probs, top3_indices = probs.topk(3)
            predicted = top3_indices[0].item()
            confidence = top3_probs[0].item() * 100

            top3_str = ", ".join(
                f"{CLASSES[i]}({p*100:.0f}%)"
                for p, i in zip(top3_probs, top3_indices)
            )

            marker = "✓" if predicted == true_label else "✗"
            print(
                f"{CLASSES[true_label]:<14} "
                f"{CLASSES[predicted]:<14} "
                f"{confidence:>9.1f}%  {top3_str}  {marker}"
            )


if __name__ == "__main__":
    predict()""",
            ),
        ],
    ),
]
