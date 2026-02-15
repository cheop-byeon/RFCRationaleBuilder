# RFCRationaleBuilder

RFCRationaleBuilder is the codebase for our curated dataset **CodeConvo**, available at https://huggingface.co/datasets/jiebi/CodeConvo/.
CodeConvo includes software engineering repositories (e.g., [nixpkgs](https://github.com/NixOS/nixpkgs), [kubernetes](https://github.com/kubernetes/kubernetes), [pytorch](https://github.com/pytorch/pytorch), [react](https://github.com/facebook/react), [rust](https://github.com/rust-lang/rust), [freecodecamp](https://github.com/freeCodeCamp/freeCodeCamp), [vscode](https://github.com/microsoft/vscode)), documentation repositories such as [kubernetes-website](https://github.com/kubernetes/website), plus [SWE-bench](https://github.com/swe-bench/SWE-bench) (converted to our formats), and many IETF WG Internet-Drafts repositories.

Two dataset variants are produced:

- **`ids`**: PR/edit ↔ issue/comment relationships from IETF draft evolution and GitHub discussions.
- **`code`**: PR/code-change ↔ issue/comment relationships from software iteration and discussions.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- For `pygit2` on macOS: `brew install libgit2`
- For `pygit2` on Ubuntu/Debian: `sudo apt-get install libgit2-dev`

### 1. Create a Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 2. Install Required Packages

```bash
# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Check that packages are installed
pip list
```

### Note: Using the Virtual Environment

Always activate the virtual environment before running scripts:

```bash
# Activate (macOS/Linux)
source venv/bin/activate

# Deactivate when done
deactivate
```

## Processing Pipeline

The full pipeline (with edge cases) is documented in [scripts/README.md](scripts/README.md).

## Dataset Citation

If you use CodeConvo or this pipeline, please cite:

```
@inproceedings{bian2024tell,
  title={Tell Me Why: Language Models Help Explain the Rationale Behind Internet Protocol Design},
  author={Bian, Jie and Welzl, Michael and Kutuzov, Andrey and Arefyev, Nikolay},
  booktitle={2024 IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN)},
  pages={447--453},
  year={2024},
  organization={IEEE}
}

@article{bian2025automated,
  title={Automated insights into github collaboration dynamics},
  author={Bian, Jie and Arefev, Nikolay and M{\"u}hlh{\"a}user, Max and Welzl, Michael},
  journal={IEEE Access},
  year={2025},
  publisher={IEEE}
}
```

## Acknowledgements

- [gh2md](https://github.com/mattduck/gh2md)
- [xml2rfc](https://github.com/ietf-tools/xml2rfc)
- [mmark](https://github.com/mmarkdown/mmark)
- [kramdown-rfc](https://github.com/gettalong/kramdown)