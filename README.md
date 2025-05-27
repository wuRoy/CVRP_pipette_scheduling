# Optimization of Robotic Liquid Handling as a Capacitated Vehicle Routing Problem

## Overview

This project implements pipette scheduling optimization for laboratory automation. It uses optimization techniques to maximize the next-tip tasks.

## Environment Setup

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository** (if applicable):
```bash
git clone <repository-url>
cd pipette_scheduling
```

2. **Create a virtual environment** (recommended, please [install ``conda``](https://conda-forge.org/download/) beforehand):
```bash
conda create --name <name-of-your-environment> python=3.9
conda activate <name-of-your-environment>

```

3. **Install required packages**:
```bash
pip install -r requirements.txt
```

4. **Install the Project in Development Mode**  

Install the package locally with:
```bash
python setup.py develop
```

## Quick Start

Because different liquid handlers use different worklist formats, we provide an interactive notebook (```pipette_scheduling/notebooks/demo.ipynb```) that allows you to specify your own output format.


## License

This project is licensed under the MIT License.
