# Toward Exploring Knowledge Graphs with LLMs
This repository contains the implementation for the poster - "[Toward Exploring Knowledge Graphs with LLMs](https://parklize.github.io/publications/Semantics2024.pdf)", [SEMANTiCS](https://semantics.cc/)'24.



## Abstract

Interacting with knowledge graphs (KGs) is challenging for non-technical users with information needs
who are unfamiliar with KG-specific query languages such as SPARQL and the underlying KG schema.
Previous KG question answering systems require ground-truth pairs of questions and queries or fine
tuning (Large) Language Models (LLMs) for a specific KG, which is time-consuming and demands deep
expertise. In this poster, we present a framework for exploring KGs for question answering using LLMs
in a zero-shot setting for non-technical end users, without the need for ground-truth pairs of questions
and queries or fine-tuning LLMs. Additionally, we evaluate an example implementation in a simple
yet challenging setting using LLMs exclusively based on the framework, without the extra effort of
maintaining the embeddings or indexes of entities from KG for retrieving relevant ones to a given
question. We share preliminary experimental results indicating that exploring a KG using LLM-generated
SPARQL queries with reasonable complexity is possible in such a challenging setting.



## Main environments

[![Python](https://img.shields.io/badge/python-3.11.0-blue?logo=python&logoColor=gold)](https://pypi.org/project/besser-agentic-framework/) 

Others can be found in ```requirements.txt```



## Folder structure


```python
├── data          # the folder contains data used for experiments
├── results       # result folder 
requirements.txt  # packages used: output from ```pip freeze > requirements.txt```
data_utils.py	  # data utils	
prompts.py	  # prompt templates 
main.py	          # main file for running experiments
```



## Citation
Guangyuan Piao, et al. "Toward Exploring Knowledge Graphs with LLMs", 20th International Conference on Semantic Systems, 2024. [[BibTex](https://parklize.github.io/bib/SEMANTICS2024.bib)]
