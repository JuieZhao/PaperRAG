# MiniRAG — 轻量级文档知识检索

> 📄 [English](README.md)

上传文档，用自然语言提问，AI 基于你的文件回答并标注来源。少模型、轻依赖、高可控。

## 为什么用 MiniRAG？

- **轻量优先** — 无需 GPU，单嵌入模型，不用 Docker，依赖极简
- **你的文档你做主** — 知识来源是你上传的文件，不是网上搜的
- **每个回答都可溯源** — 引用标注到具体文档和文本片段
- **增删文件就是增删知识库** — 完全可控
- **混合检索** — BM25 关键词 + 稠密向量，RRF 融合，召回更准
- **多格式支持** — PDF、TXT、Markdown、CSV（DOCX 即将支持）
- **内容去重** — SHA256 哈希防重复索引
- **灵活分块** — 报告按章节分，通用文档固定大小分
- **元数据过滤** — 按来源、作者、年份筛选

## 功能

- 📤 上传文档 → 自动解析、分块、嵌入、入库
- 🔀 混合检索：BM25 + 稠密向量 → RRF 融合
- 🔍 自然语言提问 → 检索 + Cross-encoder 重排序 → LLM 流式生成带引用的回答
- 🌐 中英文混合检索
- 🎯 Cross-encoder 重排序提升精度（可选）
- ⚡ 答案流式输出，边生成边显示
- 🗑️ 勾选 + 一键批量删除文档
- 📋 导出 Markdown 格式答案（含引用来源）
- 🏷️ 按作者 / 年份 / 来源筛选检索范围
- 📊 PDF 表格提取（转 Markdown，可检索）
- 🔤 嵌入模型 GPU 自动检测

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 选择前端

**Streamlit：**
```bash
streamlit run main.py
```

**Gradio（聊天式界面）：**
```bash
python gradio_app.py
```

打开浏览器对应地址，在侧边栏粘贴 DeepSeek API key，上传文件即可开始提问。

> 💡 免费获取 API key：[platform.deepseek.com](https://platform.deepseek.com)
> Key 仅保存在浏览器会话中，不会写入磁盘。
>
> 创建 `.env` 文件（参考 `.env.example`）可持久化配置。

### 3. （可选）下载文档样例

```bash
# 内置 URL 列表
python download_papers.py manual

# arXiv 关键词搜索
python download_papers.py arxiv-search "全球价值链" -c econ.GN -n 5

# arXiv RSS 最新论文
python download_papers.py arxiv-rss -c cs.AI -n 10

# 仅搜索不下载
python download_papers.py arxiv-search "supply chain resilience" --dry-run
```

## 项目结构

```
MiniRAG/
├── main.py                  # Streamlit 前端
├── gradio_app.py            # Gradio 聊天前端
├── requirements.txt
├── .env.example
├── download_papers.py       # arXiv API + RSS + 手动下载
├── test_pipeline.py         # 端到端测试
├── src/
│   ├── loader.py            # 文档解析 + 表格提取 + 分块
│   ├── embedder.py          # 文本向量化（GPU 自动检测）
│   ├── vector_store.py      # Chroma 向量库 + 元数据
│   ├── retriever.py         # 混合检索（BM25 + 稠密 + RRF）
│   ├── bm25_retriever.py    # BM25 关键词检索引擎
│   └── generator.py         # LLM 问答生成（流式）
├── data/papers/             # 您的文件（git-ignored）
└── chroma_db/               # 向量库持久化（git-ignored）
```

## 技术栈

| 层 | 技术 | 备注 |
|---|---|---|
| UI | Streamlit / Gradio | 双前端可选 |
| 解析 | PyMuPDF | PDF 按章节分块 + 表格提取 |
| 关键词检索 | BM25（自实现） | 仅依赖 numpy，无额外依赖 |
| Embedding | BGE（多模型可选） | GPU 自动检测，本地推理 |
| 向量库 | Chroma | 持久化存储，零配置 |
| LLM | DeepSeek API | 兼容 OpenAI SDK |
| 重排序 | ms-marco-MiniLM-L-6-v2 | Cross-encoder 提升精度 |
| 融合 | RRF | Reciprocal Rank Fusion |
| 来源 | arXiv API + RSS | 自动拉取文档 |

## 配置

创建 `.env` 文件：

```bash
DEEPSEEK_API_KEY=sk-xxx
EMBEDDING_MODEL=english     # 英文文档推荐
# EMBEDDING_DEVICE=cuda     # 默认自动检测
# HTTP_PROXY=http://127.0.0.1:7890  # 代理
```

## License

MIT
