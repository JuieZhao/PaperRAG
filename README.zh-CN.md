# PaperRAG — 论文知识检索库

> 📄 [English](README.md)

上传 PDF 论文，用自然语言提问，AI 基于你的文献回答并标注来源。

## 为什么用 PaperRAG？

- **你的论文你做主** — 知识来源是你上传的 PDF，不是网上搜的
- **每个回答都可溯源** — 引用标注到具体论文和文本片段
- **增删 PDF 就是增删知识库** — 完全可控

## 功能

- 📤 上传 PDF → 自动解析、分块、嵌入、入库（增量索引，不重复处理）
- 🔍 自然语言提问 → 检索 + Cross-encoder 重排序 → LLM 流式生成带引用的回答
- 🌐 中英文混合检索
- ⚡ 答案流式输出，边生成边显示
- 🗑️ 勾选 + 一键批量删除论文
- 📋 导出 Markdown 格式答案（含引用来源）
- 📊 索引进度条，处理大 PDF 不再盲目等待
- 🔍 论文列表支持搜索过滤

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动

```bash
streamlit run main.py
```

打开 http://localhost:8501 ，在侧边栏粘贴 DeepSeek API key，上传 PDF 即可开始提问。

> 💡 免费获取 API key：[platform.deepseek.com](https://platform.deepseek.com)
> Key 仅保存在浏览器会话中，不会写入磁盘。
>
> 高级用户：也可创建 `.env` 文件（参考 `.env.example`）进行文件级配置。

## 项目结构

```
PaperRAG/
├── main.py              # Streamlit 入口
├── requirements.txt
├── .env.example
├── download_papers.py   # 工具：批量下载论文
├── test_pipeline.py     # 端到端测试
├── src/
│   ├── loader.py        # PDF 解析 + 按章节结构分块
│   ├── embedder.py      # 文本向量化 (BGE 模型)
│   ├── vector_store.py  # Chroma 向量库操作
│   ├── retriever.py     # 检索 + Cross-encoder 重排序
│   └── generator.py     # LLM 问答生成（流式）
├── data/papers/         # PDF 文件（git-ignored）
└── chroma_db/           # 向量库持久化（git-ignored）
```

## 技术栈

| 层 | 技术 | 备注 |
|---|---|---|
| UI | Streamlit | `pip install streamlit` |
| PDF 解析 | PyMuPDF | 按章节/段落智能分块 |
| Embedding | BGE (BAAI/bge-small-zh-v1.5) | 本地运行，无需 API |
| 向量库 | Chroma | 持久化存储，零配置 |
| LLM | DeepSeek API | 兼容 OpenAI SDK |
| 重排序 | ms-marco-MiniLM-L-6-v2 | Cross-encoder 提升精度 |

## License

MIT
