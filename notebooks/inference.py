# %% [markdown]
# # Oreo Visibility Analysis — Gradio App
# Runtime: GPU (T4) recommended

# %% Setup
!pip install ultralytics --no-deps --quiet
!pip install "numpy>=1.23.0" "matplotlib>=3.3.0" "opencv-python-headless>=4.7.0" "pillow>=7.1.2" "pyyaml>=5.3.1" "requests>=2.23.0" "torch>=1.8.0" "torchvision>=0.9.0" "psutil>=5.8.0" "polars>=0.20.0" "nvidia-ml-py>=12.0.0" "ultralytics-thop>=2.2.3" "ultralytics-platform>=0.1.45" "huggingface_hub>=0.20" yt-dlp supabase gradio --quiet

# %% Clone repo
import sys
from pathlib import Path

REPO = "mod3-project4-anahi"
REPO_URL = f"https://github.com/Bootcamp-IA-MAD-P7/{REPO}.git"

if not Path(REPO).exists():
    !git clone --depth 1 {REPO_URL}

sys.path.insert(0, f"/content/{REPO}")

# %% Env
# Add SUPABASE_URL and SUPABASE_KEY in Colab secrets (🔑 left panel) before running this cell
import os
from google.colab import userdata

os.environ["SUPABASE_URL"] = userdata.get("SUPABASE_URL")
os.environ["SUPABASE_KEY"] = userdata.get("SUPABASE_KEY")

# %% Launch
%cd /content/{REPO}
!python app.py