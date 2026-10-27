# 🛡️ Terms & Conditions Analyzer — Real-Life Edition

**Author:** Chaitanya  
**PRN:** 124B1C108  
**Programme:** B.Tech CSE (AI & ML)  
**Project:** NLP Mini Project — Streamlit Web Application

A consumer-focused NLP application that automatically analyzes dense Terms of Service, Privacy Policies, and End-User License Agreements (EULAs) to detect hidden consumer traps, auto-billing, data selling, and rights waivers.

---

## 🌟 Key Features
- **Zero-Friction Ingestion:**
  - 🌐 Live Web URL Scraper (strips scripts, headers, navigation)
  - 📄 Multi-page PDF Contract Uploader (powered by `pypdf`)
  - ✍️ Direct text paste & built-in sample
- **⚡ TL;DR Executive Gotchas Matrix:**
  - 👁️ Data Sharing & Ad Brokers
  - 💳 Auto-Renewal & Recurring Charges
  - 🚫 Strict No-Refund Policies
  - ⚠️ Termination & Bans Without Notice
  - ⚖️ Legal Rights & Mandatory Arbitration Waivers
  - 📸 Content Takeover & Perpetual Commercial Licenses
- **💬 Hybrid Semantic Q&A:**
  - Combines TF-IDF lexical search (35%) with 50-dimensional GloVe dense vector similarity (65%) to answer questions even when user words differ from contract words.
- **✉️ Actionable Opt-Out Letter Generator:**
  - Auto-drafts legal opt-out letters for binding arbitration and data selling.
- **📥 Audit Report Export:**
  - Download findings as a Markdown (`.md`) report.

---

## 🛠️ Technology Stack
- **Frontend & App Framework:** Streamlit
- **NLP & Processing:** spaCy (`en_core_web_sm`), Gensim (`glove-wiki-gigaword-50`), NLTK
- **Machine Learning & Retrieval:** scikit-learn (Logistic Regression, TF-IDF, NMF Topic Modeling, Cosine Similarity)
- **Document & Web Parsing:** PyPDF, BeautifulSoup4, Requests
- **Visualization:** Matplotlib

---

## 🚀 Local Setup & Installation

```powershell
# 1. Clone repository
git clone https://github.com/The-Chaitanya/tc-analyzer.git
cd tc-analyzer

# 2. Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```
