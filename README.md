# Microsoft Research Internship - AI/ML Numerics & Efficiency

## Project Overview
This repository contains work for the Microsoft Research Internship take-home assignment focusing on model quantization using Microxcaling (MX) data formats on the Llama-3.2-1B model.

## 🎯 Objectives
1. **Exercise 1**: Quantize linear layers with MX (weights: mxfp4_e2m1, activations: mxfp6_e2m3)
2. **Exercise 2**: Quantize KV cache with MX (mxfp4_e2m1)
3. **Exercise 3** (Optional): Implement E5M3 scale factor support

## 📁 Repository Structure
```
msr-intern-project/
├── internexercise.txt          # Exercise instructions
├── internshipmail.txt          # Communication with Microsoft
├── scripts/                    # Setup and utility scripts
│   └── setup_colab.sh         # Colab environment setup
├── modified_files/             # Modified transformers files
│   └── modeling_llama.py      # MX-integrated Llama model
├── results/                    # Evaluation results
│   ├── baseline_results.txt
│   ├── exercise1_results.txt
│   └── exercise2_results.txt    # (pending)
└── report/                     # Final report and analysis
    └── technical_report.md
```

## 🚀 Setup Instructions

### Prerequisites
- Google Colab account with GPU access (T4/A100/H100)
- Hugging Face account with Llama-3.2-1B access
- HF access token

### Setup in Google Colab
1. Clone this repository:
```bash
git clone https://github.com/pavannn16/msr-intern-project.git
cd msr-intern-project
```

2. Run the setup script:
```bash
bash scripts/setup_colab.sh
```

3. Set your HF token:
```bash
export HF_TOKEN=<your_token_here>
```

## 📊 Timeline
- **Exercises 1 & 2 Due**: February 8, 2026, 5 PM PST
- **Exercise 3 Due**: February 13, 2026
- **Interview Window**: February 9-14, 2026
- **Baseline Completed**: January 29, 2026 ✅ (62.10% accuracy)
- **Exercise 1 Completed**: January 29, 2026 ✅ (Implementation ready)

## 🔐 Security & NDA
This repository is **PRIVATE** to comply with Microsoft NDA requirements. All work is original and properly attributed.

## 📝 Progress Tracking
- [x] Environment setup
- [x] Baseline evaluation (62.10% accuracy) ✅
- [x] Exercise 1: Linear layer quantization (implementation complete) ✅
- [x] Exercise 1: Evaluation & results (currently below target accuracy)
- [ ] Exercise 2: KV cache quantization
- [ ] Exercise 3: E5M3 scale factor (optional)
- [ ] Technical report

## 🛠️ Key Technologies
- **Model**: meta-llama/Llama-3.2-1B
- **Framework**: PyTorch, Transformers v4.57.6
- **Quantization**: Microsoft Microxcaling (MX)
- **Evaluation**: lm-eval harness (lambada_openai)
- **Platform**: Google Colab with GPU

## 📧 Contact
Pavan Chauhan - pavanc1604@gmail.com

---
*Last Updated: January 31, 2026*
