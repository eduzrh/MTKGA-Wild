# MTKGA-Wild

![Version 1.0.0](https://img.shields.io/badge/version-1.0.0-blue)
[![Language: Python 3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![Made with PyTorch](https://img.shields.io/badge/Made%20with-pytorch-orange.svg?style=flat-square)](https://www.pytorch.org/)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/eduzrh/MTKGA-Wild/issues)

[English](README.md) | [简体中文](./README_zh_CN.md)

🚀 **欢迎来到 MTKGA-Wild 仓库!** 🎉🎉🎉

本仓库包含 ICDE 2026 投稿论文的源代码: ***Taming the Wild Evolution: Aligning Multi-Modal Temporal Knowledge Graphs***.

---

## 🏠 **概览** 🔍

**野生环境下的多模态时序知识图谱对齐 (MTKGA-Wild)** 代表了一个**全新且重要的研究任务**,旨在解决**知识图谱集成**中的关键挑战。💡

### ✨ **核心创新** 🌟

#### 1. **全新且重要的研究任务** 📋

据我们所知,这是**首个**系统性探索涉及动态演化多模态信息(图像、文本、音频、视频)的多模态时序知识图谱集成的工作。我们正式提出了 **MTKGA-Wild** 问题。

#### 2. **EvoWildAlign 框架** 🔗

一个新颖的**神经符号演化智能体超图协作框架**,包含两个核心阶段:

**阶段 1: 神经符号演化超图表示** 🕵️‍♂️

通过神经检索和自适应符号解耦,自适应地将多模态时序知识图谱解耦并聚合为统一的神经符号演化超图,实现时间演化多模态事实的高质量表示。

**阶段 2: 按需智能体超图协作** 🚀

将超图推理问题转化为建模为马尔可夫决策过程的多智能体协作问题,实现自适应协调以处理不同时间点的模态可用性变化。

#### 3. **全面的基准测试** 📊

* **两个全新基准数据集**: WildMTKGA(W-I) 和 WildMTKGA(Y-I)。
* **27 种代表性基准配置**用于系统性性能评估。
* 在多样化场景下进行广泛的实验验证。

📈 通过广泛的实验验证,**EvoWildAlign** 在**多模态时序知识图谱对齐**领域建立了新的**最先进性能**,为集成演化多模态知识提供了实用范式。

---

## 🏗 **架构** 🏗️

**EvoWildAlign** 的核心架构采用**神经符号演化智能体超图协作框架**,包含两个主要阶段:

* **神经符号演化超图表示** 🕵️‍♂️: 通过神经检索和符号解耦对多模态时序知识图谱进行自适应解耦和聚合。
* **按需智能体超图协作** 🚀: 通过建模为马尔可夫决策过程的多智能体协调促进渐进式知识集成。
* **完整细节**: 请参阅论文第三节和技术报告以获取详细架构和伪代码。🔍

---

## 🔨 **主要依赖** 🛠️

* **Python** >= 3.7 (在 Python 3.8.10 上测试) 🐍
* **PyTorch** >= 1.10.0 🔥
* **Transformers** >= 4.20.0 🤖
* **SciPy** >= 1.7.0 📊
* **Pandas** >= 1.3.0 🐼
* **Tqdm** >= 4.62.0 ⏳
* **NumPy** >= 1.21.0 🔢
* **NetworkX** >= 2.6.0 🌐
* **Faiss** (用于高效相似度搜索) 🔍

---

## 📦 **安装** ⚙️

兼容 **Python 3**。🚀

1. **创建虚拟环境** (可选,但建议使用)

   ```shell
   conda create -n MTKGA-Wild python=3.8.10
   conda activate MTKGA-Wild
   ```

2. **安装依赖**

   ```shell
   pip install '主要依赖'
   ```

3. **配置 LLM API** (智能体超图协作所必需) 🔑

   配置您的 LLM API 凭据(例如 OpenAI、Claude)。示例:

   ```env
   LLM_API_KEY=your_key_here
   LLM_API_BASE=your_base_here
   LLM_MODEL=gpt-3.5-turbo-1106
   ```

---

## ✨ 数据集

数据集来自 [Dual-AMN](https://github.com/MaoXinn/Dual-AMN)、[JAPE](https://github.com/nju-websoft/JAPE)、[GCN-Align](https://github.com/1049451037/GCN-Align)、[Simple-HHEA](https://github.com/IDEA-FinAI/Simple-HHEA) 和 [BETA](https://github.com/DexterZeng/BETA)。

以数据集 icews\_wiki (HHEA) 为例,文件夹 "data/icews\_wiki" 包含:

* ent\_ids\_1: 源知识图谱中的实体 ID;
* ent\_ids\_2: 目标知识图谱中的实体 ID;
* triples\_1: 源知识图谱中由 ID 编码的关系三元组;
* triples\_2: 目标知识图谱中由 ID 编码的关系三元组;
* rel\_ids\_1: 源知识图谱中的关系 ID;
* rel\_ids\_2: 目标知识图谱中的关系 ID;
* time\_id: 源知识图谱和目标知识图谱中的时间 ID;
* ref\_ent\_ids: 所有对齐的实体对,格式为 (e\_s \t e\_t) 的对列表;

针对我们新提出的任务,我们引入了两个全新的基准数据集: **WildMTKGA(W-I)** 🌐 和 **WildMTKGA(Y-I)** 🗺️。下载链接: xxx

---

## 🔥 **快速开始** ⚡

几分钟内开始使用 **EvoWildAlign**! ⏱️

1. **克隆仓库**

   ```bash
   git clone https://github.com/eduzrh/MTKGA-Wild.git
   cd MTKGA-Wild
   ```

2. **准备数据集**

   ```bash
   # 下载并解压数据集到 ./datasets/
   ```

3. **运行主要实验**

   ```bash
   python main.py --dataset WildMTKGA(W-I)
   ```

   这将执行**完整的 EvoWildAlign 流程**:

   * 神经符号演化超图表示
   * 按需智能体超图协作
   * 实体对齐和评估

   进度通过 Tqdm 进度条监控! 📈

4. **查看结果**🔍

   * **性能指标**: 控制台输出 Hits\@1、Hits\@10 和 MRR 分数。
   * **时间/令牌消耗**: 自动计算平均处理时间和 LLM 令牌使用量。

---

## 🧑‍💻 **高级用法: 消融实验** 🔬

**EvoWildAlign** 通过灵活的组件控制支持**全面的消融实验**: 🛠️

### **消融选项**

* `--wo-neuro-symbolic`: 移除神经符号演化超图表示
* `--wo-agentic-collaboration`: 省略按需智能体超图协作
* `--wo-adaptive-decoupling`: 禁用自适应演化投影
* `--wo-core-block-selection`: 移除元智能体核心块选择
* `--wo-collaboration-decision`: 跳过协作决策
* `--wo-meta-evaluation`: 禁用元评估反馈

**示例**:

```bash
# 移除神经符号演化超图表示
python main.py --dataset W-I --wo-neuro-symbolic

# 移除按需智能体超图协作
python main.py --dataset Y-I --wo-agentic-collaboration
```

**工作流概览**:

1. **数据加载**: 从多模态时序知识图谱中提取相关实体和多模态时间事实
2. **阶段 1**: 神经符号演化超图表示

   * 神经检索 → 自适应演化投影 → 演化超图构建
3. **阶段 2**: 按需智能体超图协作

   * 核心块选择 → 协作决策 → 智能体超图执行 → 元评估
4. **评估**: 计算对齐指标并生成可视化

---

## 📊 **评估指标** 📏

我们采用标准的知识图谱对齐指标以确保**透明性和可比性**: 📐

* **Hits\@1**: 正确对齐排名第一的比例
* **Hits\@10**: 正确对齐在前 10 个候选中的比例
* **MRR (平均倒数排名)**: 正确对齐的平均倒数排名
* **效率指标**: 平均时间(秒/实体)、令牌消耗(令牌/实体)

---

## 🌍 **联系方式** 📞

📢 如有疑问或反馈,欢迎联系我们! 🙌

* 📧 **邮箱**: [runhaozhao@nudt.edu.cn](mailto:runhaozhao@nudt.edu.cn)
* 📝 **GitHub Issues**: 如有技术问题,请在 [GitHub 仓库](https://github.com/eduzrh/MTKGA-Wild/issues)中创建 Issue。标签: `bug`、`enhancement`、`question`。

目标在 **2-3 个工作日**内回复。⏱️

---

## 📜 **许可证** ⚖️

[MIT License](LICENSE) - 保留版权声明。🆓

---

## 🔗 参考文献

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

> **致谢**  ❤️
> 本工作部分参考了以下开源项目,在此表示衷心感谢:
> [Dual-AMN](https://github.com/MaoXinn/Dual-AMN)、[JAPE](https://github.com/nju-websoft/JAPE)、[GCN-Align](https://github.com/1049451037/GCN-Align)、[Simple-HHEA](https://github.com/IDEA-FinAI/Simple-HHEA)、[BETA](https://github.com/DexterZeng/BETA)、[Dual-Match](https://github.com/ZJU-DAILY/DualMatch/)、[Faiss](https://github.com/facebookresearch/faiss)、[NetworkX](https://github.com/networkx/networkx)、[AdaCoAgentEA](https://github.com/eduzrh/AdaCoAgentEA)

## **研究愉快** 🌟

**敬请关注更新!** ⭐ **收藏本仓库**以追踪我们的进展。让我们一起**驯服多模态时序知识图谱的野生演化**! ⛏️
