# MTKGA-Wild

![Version 2.0.0](https://img.shields.io/badge/version-2.0.0-blue)
[![Language: Python 3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![Made with PyTorch](https://img.shields.io/badge/Made%20with-pytorch-orange.svg?style=flat-square)](https://www.pytorch.org/)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/eduzrh/MTKGA-Wild/issues)

[English](README.md) | [简体中文](./README_zh_CN.md)

🚀 **Welcome to the MTKGA-Wild Repository!** 🎉🎉🎉

This repository contains the source code for the ICDE 2026 submission: ***Taming the Wild Evolution: Aligning Multi-Modal Temporal Knowledge Graphs***.

---

## 📰 **Latest News** 🔥

- **[Nov 14, 2025]** 🎉 Released **v2.0** with enhanced tool integration for improved usability and clearer API design!
- **[Coming Soon]** 🚧 Working on a plug-and-play toolkit version for seamless integration into existing projects.

---


## 🏠 **Overview** 🔍

**Multi-Modal Temporal Knowledge Graph Alignment in the Wild (MTKGA-Wild)** represents a **new and important research task** that addresses a critical challenge in **knowledge graph integration**.💡

### ✨ **Key Innovations** 🌟

#### 1. **New and Important Research Task** 📋

To the best of our knowledge, this is the **first work** to systematically explore the integration of MTKGs involving dynamically evolving multi-modal information such as images, text, audio, and video. We formally formulate this problem as **MTKGA-Wild**.

#### 2. **EvoWildAlign Framework** 🔗

A novel **neuro-symbolic evolutionary agentic hypergraph collaboration framework**, encompassing two core stages:

**Stage 1: Neuro-Symbolic Evolution Hypergraph Representation** 🕵️‍♂️

Adaptively decouples and aggregates MTKGs into a unified neuro-symbolic evolution hypergraph through neural retrieval and adaptive symbolic decoupling, enabling high-quality representation of temporally evolving multi-modal facts.

**Stage 2: On-Demand Agentic Hypergraph Collaboration** 🚀

Transforms the hypergraph reasoning problem into a multi-agent collaboration problem modeled as a Markov Decision Process, enabling adaptive coordination to handle varying modality availability across different time points.

#### 3. **Comprehensive Benchmarking** 📊

* **Two new benchmark datasets**: WildMTKGA(W-I) and WildMTKGA(Y-I).
* **27 representative benchmark configurations** for systematic performance evaluation.
* Extensive experimental validation across diverse scenarios.

📈 Through extensive experimental validation, **EvoWildAlign** establishes new **state-of-the-art performance** in **multi-modal temporal knowledge graph alignment**, offering a practical paradigm for integrating evolving multi-modal knowledge.

---

## 🏗 **Architecture** 🏗️

The core architecture of **EvoWildAlign** adopts a **neuro-symbolic evolutionary agentic hypergraph collaboration framework**, comprising two primary stages:

* **Neuro-Symbolic Evolution Hypergraph Representation** 🕵️‍♂️: Performs adaptive decoupling and aggregation of MTKGs through neural retrieval and symbolic decoupling.
* **On-Demand Agentic Hypergraph Collaboration** 🚀: Facilitates progressive knowledge integration via multi-agent coordination modeled as a Markov Decision Process.
* **Full Details**: Refer to Section III of the accompanying paper and the technical report for detailed architecture and pseudocode. 🔍

---

## 🔨 **Main Dependencies** 🛠️

* **Python** >= 3.7 (tested on Python 3.8.10) 🐍
* **PyTorch** >= 1.10.0 🔥
* **Transformers** >= 4.20.0 🤖
* **SciPy** >= 1.7.0 📊
* **Pandas** >= 1.3.0 🐼
* **Tqdm** >= 4.62.0 ⏳
* **NumPy** >= 1.21.0 🔢
* **NetworkX** >= 2.6.0 🌐
* **Faiss** (for efficient similarity search) 🔍

---

## 📦 **Installation** ⚙️

Compatible with **Python 3**. 🚀

1. **Create a Virtual Environment** (optional, but recommended)

   ```shell
   conda create -n MTKGA-Wild python=3.8.10
   conda activate MTKGA-Wild
   ```

2. **Install Dependencies**

   ```shell
   pip install 'Main Dependencies'
   ```

3. **Configure LLM API** (required for agentic hypergraph collaboration) 🔑

   Configure your LLM API credentials (e.g., OpenAI, Claude). Example:

   ```env
   LLM_API_KEY=your_key_here
   LLM_API_BASE=your_base_here
   LLM_MODEL=gpt-3.5-turbo-1106
   ```

---

## ✨ Datasets

The datasets are from [Dual-AMN](https://github.com/MaoXinn/Dual-AMN), [JAPE](https://github.com/nju-websoft/JAPE), [GCN-Align](https://github.com/1049451037/GCN-Align), [Simple-HHEA](https://github.com/IDEA-FinAI/Simple-HHEA) and [BETA](https://github.com/DexterZeng/BETA).

Take the dataset icews\_wiki (HHEA) as an example, the folder "data/icews\_wiki" contains:

* ent\_ids\_1: ids for entities in source KG;
* ent\_ids\_2: ids for entities in target KG;
* triples\_1: relation triples encoded by ids in source KG;
* triples\_2: relation triples encoded by ids in target KG;
* rel\_ids\_1: relation ids in the source KG;
* rel\_ids\_2: relation ids in the target KG;
* time\_id: time ids in the source KG and the target KG;
* ref\_ent\_ids: all aligned entity pairs, list of pairs like (e\_s \t e\_t);

For our newly proposed task, we introduce two novel benchmark datasets: **WildMTKGA(W-I)** 🌐 and **WildMTKGA(Y-I)** 🗺️. Download link: xxx

---

## 🔥 **Quick Start** ⚡

Get started with **EvoWildAlign** in minutes! ⏱️

1. **Clone the Repository**

   ```bash
   git clone https://github.com/eduzrh/MTKGA-Wild.git
   cd MTKGA-Wild
   ```

2. **Prepare Datasets**

   ```bash
   # Download and extract datasets to ./datasets/
   ```

3. **Run the Main Experiment**

   ```bash
   python main.py --dataset WildMTKGA(W-I)
   ```

   This executes the **complete EvoWildAlign pipeline**:

   * Neuro-symbolic evolution hypergraph representation
   * On-demand agentic hypergraph collaboration
   * Entity alignment and evaluation

   Progress is monitored via Tqdm progress bars! 📈

4. **View Results**🔍

   * **Performance Metrics**: Console outputs Hits\@1, Hits\@10, and MRR scores.
   * **Time/Token Consumption**: Automatically calculates average processing time and LLM token usage.

---

## 🧑‍💻 **Advanced Usage: Ablation Studies** 🔬

**EvoWildAlign** supports **comprehensive ablation experiments** via flexible component control: 🛠️

### **Ablation Options**

* `--wo-neuro-symbolic`: Removes neuro-symbolic evolution hypergraph representation
* `--wo-agentic-collaboration`: Omits on-demand agentic hypergraph collaboration
* `--wo-adaptive-decoupling`: Disables adaptive evolution projection
* `--wo-core-block-selection`: Removes meta-agent core block selection
* `--wo-collaboration-decision`: Skips collaboration decision-making
* `--wo-meta-evaluation`: Disables meta evaluation feedback

**Example**:

```bash
# Remove neuro-symbolic evolution hypergraph representation
python main.py --dataset W-I --wo-neuro-symbolic

# Remove on-demand agentic hypergraph collaboration
python main.py --dataset Y-I --wo-agentic-collaboration
```

**Workflow Overview**:

1. **Data Loading**: Extract relevant entities and multi-modal temporal facts from MTKGs
2. **Stage 1**: Neuro-symbolic evolution hypergraph representation

   * Neural retrieval → Adaptive evolution projection → Evolution hypergraph construction
3. **Stage 2**: On-demand agentic hypergraph collaboration

   * Core block selection → Collaboration decision-making → Agentic hypergraph execution → Meta evaluation
4. **Evaluation**: Compute alignment metrics and generate visualizations

---

## 📊 **Evaluation Metrics** 📏

We employ standard knowledge graph alignment metrics for **transparency and comparability**: 📐

* **Hits\@1**: Proportion of correct alignments ranked first
* **Hits\@10**: Proportion of correct alignments in top-10 candidates
* **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of correct alignments
* **Efficiency Metrics**: Average time (seconds/entity), token consumption (tokens/entity)

---

## 🌍 **Contact Information** 📞

📢 For inquiries or feedback, we welcome your contact! 🙌

* 📧 **Email**: [runhaozhao@nudt.edu.cn](mailto:runhaozhao@nudt.edu.cn)
* 📝 **GitHub Issues**: For technical concerns, create an Issue in the [GitHub repository](https://github.com/eduzrh/MTKGA-Wild/issues). Labels: `bug`, `enhancement`, `question`.

Responses targeted within **2-3 business days**. ⏱️

---

## 📜 **License** ⚖️

[MIT License](LICENSE) - Copyright notices preserved. 🆓

---

## 🔗 References

* [Unsupervised Entity Alignment for Temporal Knowledge Graphs](https://doi.org/10.1145/3543507.3583381).
  Xiaoze Liu, Junyang Wu, Tianyi Li, Lu Chen, and Yunjun Gao.
  Proceedings of the ACM Web Conference (WWW), 2023.
* [BERT-INT: A BERT-based Interaction Model for Knowledge Graph Alignment](https://doi.org/10.1145/3543507.3583381).
  Xiaobin Tang, Jing Zhang, Bo Chen, Yang Yang, Hong Chen, and Cuiping Li.
  Journal of Artificial Intelligence Research, 2020.
* [Benchmarking Challenges for Temporal Knowledge Graph Alignment](https://api.semanticscholar.org/CorpusID:273501043).
  Weixin Zeng, Jie Zhou, and Xiang Zhao.
  Proceedings of the ACM International Conference on Information and Knowledge Management (CIKM), 2024.
* [Cross-lingual Knowledge Graph Alignment via Graph Convolutional Networks](https://doi.org/10.18653/v1/d18-1032).
  Zhichun Wang, Qingsong Lv, Xiaohan Lan, and Yu Zhang.
  Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018.
* [Boosting the Speed of Entity Alignment 10×: Dual Attention Matching Network with Normalized Hard Sample Mining](https://doi.org/10.1145/3442381.3449897).
  Xin Mao, Wenting Wang, Yuanbin Wu, and Man Lan.
  Proceedings of the Web Conference (WWW), 2021.
* [Wikidata: A Free Collaborative Knowledgebase](https://doi.org/10.1145/2629489).
  Denny Vrandecic and Markus Krötzsch.
  Communications of the ACM, 2014.
* [Toward Practical Entity Alignment Method Design: Insights from New Highly Heterogeneous Knowledge Graph Datasets](https://doi.org/10.1145/3589334.3645720).
  Xuhui Jiang, Chengjin Xu, Yinghan Shen, Yuanzhuo Wang, Fenglong Su, Zhichao Shi, Fei Sun, Zixuan Li, Jian Guo, and Huawei Shen.
  Proceedings of the ACM Web Conference (WWW), 2024.
* [Unlocking the Power of Large Language Models for Entity Alignment](https://aclanthology.org/2024.acl-long.408).
  Xuhui Jiang, Yinghan Shen, Zhichao Shi, Chengjin Xu, Wei Li, Zixuan Li, Jian Guo, Huawei Shen, and Yuanzhuo Wang.
  Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), 2024.
* [Bootstrapping Entity Alignment with Knowledge Graph Embedding](https://doi.org/10.24963/ijcai.2018/611).
  Zequn Sun, Wei Hu, Qingheng Zhang, and Yuzhong Qu.
  Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 2018.
* [NetworkX: Network Analysis in Python](https://github.com/networkx/networkx).
  NetworkX Developers.
  GitHub Repository.
* [Faiss: A Library for Efficient Similarity Search and Clustering of Dense Vectors](https://github.com/facebookresearch/faiss).
  Facebook Research.
  GitHub Repository.

> **Acknowledgement**  ❤️
> The following open source projects were partially referenced in this work. We sincerely appreciate their contributions:
> [Dual-AMN](https://github.com/MaoXinn/Dual-AMN), [JAPE](https://github.com/nju-websoft/JAPE), [GCN-Align](https://github.com/1049451037/GCN-Align), [Simple-HHEA](https://github.com/IDEA-FinAI/Simple-HHEA), [BETA](https://github.com/DexterZeng/BETA), [Dual-Match](https://github.com/ZJU-DAILY/DualMatch/), [Faiss](https://github.com/facebookresearch/faiss), [NetworkX](https://github.com/networkx/networkx), [AdaCoAgentEA](https://github.com/eduzrh/AdaCoAgentEA)

## **Happy Researching** 🌟

**Stay tuned for updates!** ⭐ **Star this repository** to track our progress. Let's **tame the wild evolution** of multi-modal temporal knowledge graphs together! ⛏️
