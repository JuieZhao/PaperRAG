# PaperRAG — 论文知识检索库

> 📄 [English](README.md)

上传 PDF 论文，用自然语言提问，AI 基于你的文献回答并标注来源。

## 为什么用 PaperRAG？

- **你的论文你做主** — 知识来源是你上传的 PDF，不是网上搜的
- **每个回答都可溯源** — 引用标注到具体论文和文本片段
- **增删 PDF 就是增删知识库** — 完全可控
- **混合检索（新）** — BM25 关键词 + 稠密向量，RRF 融合，召回更准
- **内容去重（新）** — SHA256 哈希防重复索引
- **表格提取（新）** — PDF 表格转 Markdown，可检索
- **元数据过滤（新）** — 按作者、年份、论文名筛选

## 功能

- 📤 上传 PDF → 自动解析、分块、嵌入、入库
- 🔀 混合检索：BM25 + 稠密向量 → RRF 融合
- 🔍 自然语言提问 → 检索 + Cross-encoder 重排序 → LLM 流式生成带引用的回答
- 🌐 中英文混合检索
- 🎯 Cross-encoder 重排序提升精度
- ⚡ 答案流式输出，边生成边显示
- 🗑️ 勾选 + 一键批量删除论文
- 📋 导出 Markdown 格式答案（含引用来源）
- 🏷️ 按作者 / 年份 / 论文名筛选检索范围
- 📊 PDF 表格提取（转 Markdown，可检索）
- 🔤 嵌入模型 GPU 自动检测

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 选择前端

**Streamlit（经典版）：**
```bash
streamlit run main.py
```

**Gradio（新 — 聊天式界面）：**
```bash
python gradio_app.py
```

打开浏览器对应地址，在侧边栏粘贴 DeepSeek API key，上传 PDF 即可开始提问。

> 💡 免费获取 API key：[platform.deepseek.com](https://platform.deepseek.com)
> Key 仅保存在浏览器会话中，不会写入磁盘。
>
> 创建 `.env` 文件（参考 `.env.example`）可持久化配置。

### 3. （可选）下载论文样例

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
PaperRAG/
├── main.py                  # Streamlit 前端
├── gradio_app.py            # Gradio 聊天前端（新）
├── requirements.txt
├── .env.example
├── download_papers.py       # arXiv API + RSS + 手动下载
├── test_pipeline.py         # 端到端测试
├── src/
│   ├── loader.py            # PDF 解析 + 表格提取 + 分块
│   ├── embedder.py          # 文本向量化（GPU 自动检测）
│   ├── vector_store.py      # Chroma 向量库 + 元数据
│   ├── retriever.py         # 混合检索（BM25 + 稠密 + RRF）
│   ├── bm25_retriever.py    # BM25 关键词检索引擎（新）
│   └── generator.py         # LLM 问答生成（流式）
├── data/papers/             # PDF 文件（git-ignored）
└── chroma_db/               # 向量库持久化（git-ignored）
```

## 技术栈

| 层 | 技术 | 备注 |
|---|---|---|
| UI | Streamlit / Gradio | 双前端可选 |
| PDF 解析 | PyMuPDF | 按章节分块 + 表格提取 |
| 关键词检索 | BM25（自实现） | 仅依赖 numpy，无额外依赖 |
| Embedding | BGE（多模型可选） | GPU 自动检测，本地推理 |
| 向量库 | Chroma | 持久化存储，零配置 |
| LLM | DeepSeek API | 兼容 OpenAI SDK |
| 重排序 | ms-marco-MiniLM-L-6-v2 | Cross-encoder 提升精度 |
| 融合 | RRF | Reciprocal Rank Fusion |
| 论文来源 | arXiv API + RSS | 自动拉取新论文 |

## 配置

创建 `.env` 文件：

```bash
DEEPSEEK_API_KEY=sk-xxx
EMBEDDING_MODEL=english     # 英文论文推荐
# EMBEDDING_DEVICE=cuda     # 默认自动检测
# HTTP_PROXY=http://127.0.0.1:7890  # 代理
```

## License

MIT
