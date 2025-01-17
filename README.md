# LangGraph Memory System

## 🧠 智能生成式AI记忆增强系统

![GitHub stars](https://img.shields.io/github/stars/yourusername/langgraph-memory-system?style=social)
![Python Version](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

一个基于LangGraph和LLM技术栈的多层次记忆管理系统，解决大型语言模型（LLMs）的上下文限制问题，实现长期记忆、信息检索与生成式AI的统一。该系统支持跨会话记忆保持、基于相关性的历史信息检索以及事件驱动的记忆流水线处理。

## 🌟 核心特性

- **多层次记忆架构**：集成短期记忆（Redis）和长期记忆（MongoDB向量存储）的混合记忆模型
- **基于LangGraph的工作流引擎**：使用DAG定义记忆管理工作流，实现复杂记忆处理流程
- **语义化记忆检索**：使用向量相似度搜索实现基于上下文的智能记忆召回
- **事件驱动架构**：通过Kafka实现高性能、可扩展的事件总线，支持异步记忆处理
- **高性能分布式推理**：基于Ray和vLLM的水平扩展推理服务，支持大规模模型部署
- **云原生部署**：完整的Kubernetes配置，支持生产环境的弹性伸缩

## 🔧 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| **核心框架** | [LangGraph](https://github.com/langchain-ai/langgraph) | 记忆工作流管理 |
|  | [LangChain](https://github.com/hwchase17/langchain) | LLM应用开发框架 |
| **存储层** | [Redis](https://redis.io/) | 短期记忆、缓存服务 |
|  | [MongoDB Atlas](https://www.mongodb.com/atlas/database) | 长期记忆存储、向量搜索 |
| **消息队列** | [Apache Kafka](https://kafka.apache.org/) | 事件总线、异步处理 |
| **推理服务** | [vLLM](https://github.com/vllm-project/vllm) | 高性能LLM推理引擎 |
|  | [Ray](https://www.ray.io/) | 分布式计算框架 |
| **API服务** | [FastAPI](https://fastapi.tiangolo.com/) | 高性能REST API |
| **部署平台** | [Kubernetes](https://kubernetes.io/) | 容器编排、云部署 |
|  | [Docker](https://www.docker.com/) | 容器化 |

## 🏗️ 系统架构

```mermaid
graph TD
    A[用户输入] --> B[API层 FastAPI]
    B --> C[LangGraph工作流]
    C --> D[记忆检索]
    C --> E[响应生成]
    C --> F[记忆更新]
    
    D --> G[短期记忆 Redis]
    D --> H[长期记忆 MongoDB]
    
    E --> I[推理服务 vLLM/Ray]
    
    F --> J[事件总线 Kafka]
    J --> K[异步记忆处理]
    K --> H
    
    style C fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#bbf,stroke:#333
    style H fill:#bbf,stroke:#333
    style I fill:#bfb,stroke:#333
    style J fill:#fbb,stroke:#333 


## 记忆系统工作流程

sequenceDiagram
    participant Client
    participant API
    participant LangGraph
    participant ShortTerm
    participant LongTerm
    participant LLM
    
    Client->>API: 发送用户消息
    API->>LangGraph: 调用记忆工作流
    LangGraph->>ShortTerm: 检索近期对话
    LangGraph->>LongTerm: 检索相关记忆
    ShortTerm-->>LangGraph: 返回对话历史
    LongTerm-->>LangGraph: 返回相关记忆
    LangGraph->>LLM: 组装上下文发送到LLM
    LLM-->>LangGraph: 生成响应
    LangGraph->>ShortTerm: 保存新消息
    LangGraph->>LongTerm: 保存重要记忆
    LangGraph-->>API: 返回结果
    API-->>Client: 返回响应