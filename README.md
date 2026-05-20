# 🌐 Global Affairs Regulatory Intelligence Pipeline (GARIP)

An end-to-end ETL + RAG system that ingests multi-source regulatory and policy data, surfaces jurisdiction-level compliance signals, and connects policymakers to an LLM-powered advisory interface.

## 📌 Overview
GARIP is production-grade data engineering project built to demonstrate skills directly aligned with Google's Global Affairs (GA) Business Intelligence Group. It simulates the kind of infrasturucture needed to move, transform, and intelligently query regulatory data across multiple jurisdictions — enabling data-driven, AI-assisted policy decisions at scale.

The system ingests publicly available regulatory documents from sources like EUR-Lex, Congress.giv, FTC, and regulations.gov, processes them through a structured ETL layer, stores them in a dual-store architecture (BigQuery-equivalent + vector database), and serves a policymaker-facing Q&A powered by a Large Language Model API.

## 🏗️ Architecture
<img width="374" height="626" alt="image" src="https://github.com/user-attachments/assets/5e4301a0-fd46-4f24-a451-40f07f7ce1df" />

## ✨ Key Features
**🔄 Multi-Source ETL Pipeline**
- Source-specific Python connectors for 5 public regulatory APIs and document feeds
- Incremental load pattern with version_id + source_hash to track regulatory amendments without overwriting history
- Schema drift detection and null-rate alerts logged to a pipeline_health table

**📄 Structure-Aware Document Processing**
- PDF parsing with pdfplumber / pymupdf that preserves hierarchical structure (Articles -> Clauses -> Subsections)
- Each chunk retains metadata: article_id, jurisdiction, regulation_name, effective_date, regulation_type
- 512-token chunks with 10% overlap, optimized for regulatory document length distributions

**🔍 Hybrid Retrieval (RAG)**
- **Dense Retrieval:** Sentence-transformer embeddings indexed in Pinecone / ChromaDB
- **Sparse interval:** BM25 for exact legal terminology matching (e.g., "Section 230", Article 12 DSA")
- **Reranking:** Cross-encoder reranker for precision on jurisdiction-specif queries
- **Jurisdiction-aware filtering:** Metadata filters ensure responses are scoped to relevant legal contexts

**🤖 LLM-Powered Advisory Interface**
- Natural language Q&A over the regulatory corpus via Claude / Gemini API
- Responses include source citations with document name, jurisdiction, and article reference
- Conflict detection: flags when two jurisdictions have contradictory rules on the same topic

**📊 Pipeline Observability**
- Every ETL run logs row counts, schema drift, null rates, and deduplication scores
- Data quality scorecard rendered in Streamlit
- Dagster orchestration with alerting on SLA breaches

## 🗂️ Project Structure
<img width="374" height="822" alt="image" src="https://github.com/user-attachments/assets/6412adf4-f030-4ac5-87d2-6bd2f568189d" />

## 🛠️ Tech Stack
<img width="624" height="472" alt="image" src="https://github.com/user-attachments/assets/0b6bd39f-4c71-48b3-b460-26f7b2c502aa" />

## 📦 Data Sources
All data sources are publicly available —— no licenses or access fees required.
<img width="653" height="232" alt="image" src="https://github.com/user-attachments/assets/c2f09426-c3d1-4867-8f99-a1cf31ccec6a" />

## 🚀 Getting Started
**Prerequisites**
- Python 3.11+
- Docker + Docker Compose
- Pinecone API key (free tier sufficient)
- Anthropic API key (Claude) or Google AI API key (Gemini)
- Congress.gov API key (free registration)

Installation
### Clone the repository
git clone https://github.com/sajansshergill/garip.git
cd garip

### Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

### Install dependencies
pip install -r requirements.txt

### Copy and configure environment variables
cp .env.example .env
### Edit .env with your API keys

**Environment Variables**
### .env.example
ANTHROPIC_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
PINECONE_ENV=us-east-1
CONGRESS_API_KEY=your_key_here
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=claude-sonnet-4-20250514
DUCKDB_PATH=./data/garip.duckdb

**Running the Pipeline**
### Run full ingestion + ETL via Dagster
dagster dev -f orchestration/dagster_pipeline.py

### Or run individual connectors directly
python ingestion/eurlex_connector.py --limit 100
python etl/pdf_parser.py --source ico
python storage/vector_loader.py --batch-size 50

### Launch Streamlit app
streamlit run app/app.py

## 📊 Data Model

regulations (DuckDB / BigQuery)
<img width="620" height="399" alt="image" src="https://github.com/user-attachments/assets/b5d5ffdc-dfda-48fb-b3cf-32c958eb8a4d" />

pipeline_health (DuckDB / BigQuery)
<img width="576" height="370" alt="image" src="https://github.com/user-attachments/assets/0e3da2b3-7b99-4590-8796-c59701d143a5" />

## 🧪 Testing
bash# Run full test suite
pytest tests/ -v

### Run with coverage report
pytest tests/ --cov=garip --cov-report=html

### Run specific module
pytest tests/test_retrieval.py -v

## 🔭 Stretch Goals
- **Agentic monitoring:** LangGraph agent that autonomously watches for new regulatory filings, triggers ingestion, and sends a daily ingest summarizing changes
- **Conflict detection dashboard:** SQL queries that surface contradictions between jurisdictions on the same regulatory topic (e.g., EU data localization vs. US cloud subpoena requirements)
- **Live pipeline stress test:** SQL queries that surface contradictions between jurisdictions on the same regulatory topic (e.g., EU data localization vs. US cloud subpoena requirements)
- **GCP migration guide:** Step-by-step mapping of local stack to full GCP deployment (DuckDB → BigQuery, Pinecone → Vertex AI Vector Search, Dagster → Cloud Composer, Streamlit → Cloud Run)

## 📈 Skills Demonstrated
<img width="614" height="419" alt="image" src="https://github.com/user-attachments/assets/fb7b4652-63e5-4d05-b9c8-7f3eb4fdab66" />

## 👤 Author
**Sajan Shergill** M.S. Data Science — Pace University, Seidenberg School (2026)
linkedin.com/in/sajanshergill · sajansshergill.github.io · (551) 358-4302


