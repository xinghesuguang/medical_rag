# 医疗RAG知识库问答系统

## 项目简介

这是一个基于LangChain和微调模型的医疗领域RAG（检索增强生成）问答系统。系统集成了DeepSeek-R1模型的SFT和DPO训练，提供专业的医疗知识问答服务。

## 主要特性

- 🤖 智能医疗问答系统
- 📚 支持多种文档格式（txt、docx、json）
- 🔧 集成SFT和DPO模型微调
- 🌐 基于Gradio的Web界面
- 🚀 本地化部署，数据安全
- 📊 FAISS向量检索

## 项目结构

```
medical_rag/
├── README.md                    # 项目说明
├── rag_gradio_enhanced.py       # 主程序
├── file_t/                      # 知识库文档
│   ├── medical_book_zh.json
│   ├── test_encyclopedia.txt
│   └── valid_encyclopedia.docx
└── train_file/                  # 训练相关
    ├── data/                    # 训练数据
    ├── deepseekr1_7B_lora_sft.yaml
    ├── deepseekr1_7B_lora_dpo.yaml
    ├── result_sft/              # SFT结果
    └── result_dpo/              # DPO结果
```

## 快速开始

### 环境要求
- Python 3.8+
- CUDA 11.8+
- 显存 >= 16GB

### 安装依赖
```bash
pip install langchain langchain-community langchain-huggingface
pip install transformers gradio faiss-cpu python-docx
```

### 运行系统
```bash
python rag_gradio_enhanced.py
```

访问 `http://localhost:7862` 使用Web界面。

## 使用说明

1. **初始化系统**: 点击"初始化系统"按钮
2. **添加文档**: 将文档放入 `file_t/` 目录
3. **开始问答**: 在界面中输入医疗问题

### 示例问题
- "宫颈口粘连怎么回事？"
- "前列腺炎应该吃什么食物比较好？"
- "中风有什么症状？"

## 模型训练

### SFT训练
```bash
llamafactory-cli train train_file/deepseekr1_7B_lora_sft.yaml
```

### DPO训练
```bash
llamafactory-cli train train_file/deepseekr1_7B_lora_dpo.yaml
```

## 技术栈

- **框架**: LangChain, Transformers
- **模型**: DeepSeek-R1, BGE-Large-ZH
- **向量库**: FAISS
- **界面**: Gradio
- **训练**: LLaMA-Factory, DeepSpeed

## 注意事项

- 确保模型路径配置正确
- 建议使用24GB+显存的GPU
- 支持UTF-8和GBK编码
- 仅供学习研究使用

## 许可证

MIT License