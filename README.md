# PaperRAG — 论文知识检索库

> 让用户上传自己的 PDF 论文，用自然语言提问，AI 基于文献回答并标注来源。

## 核心理念
- **不是通用聊天机器人**：知识来源是你自己上传的论文，不是网上搜的
- **每个回答都可溯源**：引用标注到具体论文和片段
- **你的论文你做主**：增删 PDF 就是增删知识库

## 核心功能
- 上传 PDF 论文 → 自动解析、向量化、入库
- 自然语言提问 → 检索相关片段 → LLM 生成带引用的回答
- 支持中英文混合检索

---

## 项目结构

```
PaperRAG/
├── README.md
├── requirements.txt
├── .env.example
├── main.py              # Streamlit 入口
├── src/
│   ├── __init__.py
│   ├── loader.py        # PDF 解析 + 文本分块
│   ├── embedder.py      # 文本向量化
│   ├── vector_store.py  # Chroma 向量库操作
│   ├── retriever.py     # 检索逻辑
│   └── generator.py     # LLM 问答生成
├── data/
│   └── papers/          # 放你要索引的 PDF
└── chroma_db/           # 向量库持久化存储
```

## Tech Stack
| 层 | 技术 | 备注 |
|---|---|---|
| UI | Streamlit | `pip install streamlit` |
| PDF 解析 | PyMuPDF (fitz) | `pip install pymupdf` |
| 文本分块 | LangChain RecursiveCharacterTextSplitter | 按语义边界切分 |
| Embedding | text2vec-large-chinese 或 BGE | 中文友好 |
| 向量库 | Chroma | `pip install chromadb` |
| LLM | DeepSeek API | `pip install openai`，兼容 OpenAI SDK |
