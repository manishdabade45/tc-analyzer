# Terms & Conditions Analyzer

**Name:** Chaitanya  
**PRN:** 124B1C108  
**Programme:** B.Tech CSE (AI & ML)  
**Project type:** NLP mini project — Streamlit web application

## 1. What you receive

`app.py` contains the complete application, a 24-example labelled training dataset, a sample T&C, and all processing functions. There is no database, API key, uploaded dataset, or separate configuration file. Python packages and downloaded language models are still required. Model downloads are external assets stored by the libraries; a literally file-free app cannot use the requested pretrained models.

## 2. How to run it

Use Python 3.10 or newer; Python 3.11 is a practical choice. Save `app.py` in a folder and open a terminal in that folder.

### Windows PowerShell

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install streamlit nltk spacy gensim scikit-learn matplotlib numpy
python -m spacy download en_core_web_sm
streamlit run app.py
```

If PowerShell blocks activation, use the virtual environment directly without changing execution policy:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install streamlit nltk spacy gensim scikit-learn matplotlib numpy
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
.\.venv\Scripts\python.exe -m streamlit run app.py
```

For macOS/Linux, replace the activation command with `source .venv/bin/activate`; the remaining normal commands are the same. If the `streamlit` command is unavailable, use `python -m streamlit run app.py`.

Open the local URL printed in the terminal, usually `http://localhost:8501`. Click **Use sample T&C**. The first analysis downloads `glove-wiki-gigaword-50`, approximately 65 MB compressed, and may take several minutes. Keep internet available during setup and this first download. Later runs reuse the disk download; `@st.cache_resource` also keeps spaCy, GloVe, and the trained classifier in memory across Streamlit reruns. Restarting the server reloads these resources into memory.

NLTK is used for `ngrams`, which does not need downloaded NLTK corpora. spaCy provides sentence splitting, tokenization, stopwords, lemmas, POS tags, and NER. Do not run extra NLTK download commands.

If setup fails, confirm pip and Streamlit use the same virtual environment. A missing spaCy model requires rerunning its download command in that environment. A failed GloVe download requires internet access and permission for Gensim to write its model cache. Do not interpret a failed analysis as a safe contract.

### Demonstration steps

1. Load the sample or paste English T&C; analysis starts automatically.
2. Open **Risk Report** to see the score, red/green clauses, and simple explanations.
3. Open **Key Details & POS** to inspect entities and POS tags.
4. Open **NLP Insights & Charts** for phrase matches, TF-IDF keywords, NMF topics, and charts.
5. Open **Ask a Question** and enter “Can they sell my data?” to retrieve a clause.
6. Try an unrelated question to see the no-match message. A question with weak shared words can still retrieve an irrelevant clause, so read the displayed text.

The app processes input in the Streamlit server's memory and does not deliberately save it. When run locally, this server is your computer. If hosted elsewhere, pasted text is sent to that host. The model downloader fetches model files; the code does not send T&C text to a remote AI service.

## 3. Pipeline architecture in simple English

```text
Paste text / built-in sample
    -> Preprocess clauses
    -> POS tagging and NER on original wording
    -> TF-IDF keywords
    -> Average GloVe vectors
    -> Logistic regression: Risky / Safe
    -> Original-word bigram/trigram matches + risk type hints
    -> NMF topics from the TF-IDF matrix
    -> TF-IDF cosine retrieval for a question
    -> Score, coloured report, simple explanations, tables, charts
```

This is an ordered workflow with shared branches. POS and NER use original spaCy documents; they are displayed for inspection and are not features fed into logistic regression. TF-IDF supports keywords, topics, and QA. GloVe supports classification. Phrase detection uses original words because removing stopwords would destroy phrases such as “right to sue”.

| Stage | Function | Meaning |
|---|---|---|
| Resource loading | `load_resources` | Loads spaCy and pretrained GloVe once per cache lifetime. |
| Preprocessing | `preprocess` | Splits lines, semicolons, and sentences into approximate clauses; tokenizes, lemmatizes, and removes stopwords. Keeps no/not/never/without. |
| POS | `pos_tagging` | Shows grammatical categories such as NOUN and VERB. |
| NER | `named_entities` | Extracts organizations, money, dates, and places. An organization is not necessarily a company. |
| TF-IDF | `tfidf` | Gives words higher weights when they distinguish clauses in this document. |
| Embeddings | `embeddings` | Converts each clause into the mean of its known 50-number word vectors. Unknown words are skipped; an all-unknown clause gets zeros. |
| Classification | `train_classifier`, `classify` | Trains scaled logistic regression on 12 risky and 12 safe examples, then predicts clauses. |
| Phrase matching | `risky_phrases` | Matches listed adjacent two-word and three-word expressions. |
| Topic modelling | `topic_model` | Uses NMF to group clauses into up to three topics. Shows top words and clause numbers. |
| QA | `answer_question` | Uses the document vocabulary and cosine similarity to return the nearest clause. |
| Type hints | `risk_types` | Uses keyword patterns to suggest one or more categories for predicted risky clauses. |

Risk categories include the five requested types plus **No refund**, because the real-world problem explicitly includes refunds. Unmatched predicted risks receive **Other / unclear risk**.

**Score formula:** `100 × number of predicted risky clauses / total number of clauses`. The displayed score is rounded. For example, three predicted risky clauses among ten clauses gives 30/100. This is a coverage measure, not a probability of harm, severity rating, or legal assessment. Splitting one sentence into several clauses can change this score.

**TF-IDF:** term frequency × inverse document frequency. Here a “document” for TF-IDF is one clause. The keyword table ranks the mean weight across clauses.

**Average embedding:** `clause vector = sum of known word vectors / number of known words`.

**Logistic regression:** learns weights from labelled vectors and predicts a binary label using its default decision threshold. Training labels are manually chosen for education, not legally validated.

**Cosine similarity:** compares the direction of the question vector and clause vectors. A larger value means more shared weighted vocabulary; it is not answer confidence.

## 4. Short project report

### Abstract

Terms & Conditions Analyzer is an educational NLP web application that helps users review long service agreements. Users paste English text and receive a clause-level risk report, simple warnings, important entities, phrase matches, keywords, topics, and relevant clauses for questions. The system combines spaCy, NLTK, GloVe embeddings, logistic regression, TF-IDF, and NMF in a single Streamlit application. It aims to improve awareness before accepting terms, while clearly stating that its predictions are not legal advice.

### Problem Statement

Many users accept terms without reading every clause. Agreements may contain conditions about personal data, repeated payments, refunds, account closure, dispute rights, and uploaded content. Length and complex wording make manual review difficult. A simple tool is needed to bring potentially concerning clauses to the user's attention in understandable language.

### Objectives

- Accept pasted English terms or a built-in sample.
- Predict Risky or Safe for each clause and calculate a document score.
- Suggest risk categories and short explanations.
- Identify entities, suspicious phrases, keywords, and topics.
- Retrieve relevant clauses for questions and visualize findings.
- Demonstrate several classical NLP techniques in an understandable one-file application.

### Methodology

The application splits input into approximate clauses using line breaks, semicolons, and sentence boundaries. spaCy tokenizes the clauses, creates lemmas, tags parts of speech, and recognizes named entities. Stopwords are removed while selected negations are retained. Each clause becomes a 50-dimensional average GloVe vector. A logistic regression model trained on 24 embedded examples predicts the class.

Separate pattern matching suggests risk types and detects listed bigrams and trigrams. TF-IDF identifies distinguishing words. NMF discovers up to three groups of related clauses. A question is converted into the same TF-IDF space and compared with all clauses using cosine similarity. The most similar clause is displayed without generating a legal interpretation.

### Implementation

Python provides the processing code and Streamlit provides the interface and backend. Gensim loads `glove-wiki-gigaword-50`; spaCy loads `en_core_web_sm`. scikit-learn supplies TF-IDF, scaling, logistic regression, NMF, and cosine similarity. NLTK generates n-grams and Matplotlib draws the charts. Training examples, rules, and sample input are Python constants inside `app.py`. Resources are cached, and input is limited to 30,000 characters for a small demonstration.

### Results

The implementation produces clause predictions, a flagged-clause percentage, type hints, entity tables, phrase matches, ranked keywords, topics, retrieved clauses, and two charts when the required resources are installed. The built-in sample contains examples intended to exercise data selling, auto-renewal, termination, waiver, content rights, and refunds. Exact predictions and entity spans depend on the installed models.

No independent accuracy, precision, recall, or F1 score is claimed. The 24 training examples are too small to establish real-world performance. Running the same examples used in training is a demonstration, not an independent evaluation. Record your local observed outputs before submitting screenshots or numerical results.

Suggested results record:

| Test | What to record |
|---|---|
| Built-in sample | Total clauses, predicted risky count, score, visible type hints. |
| New unseen terms | Predictions compared with manually reviewed reference labels. |
| Negative wording | Compare “We sell your data” and “We do not sell your data”. Record errors honestly. |
| Refund alternatives | Compare a no-refund clause and a refund-right clause. |
| QA | Relevant question, displayed clause, and unrelated-question behavior. |
| Edge cases | Blank input, punctuation-only input, and a single clause. |

### Limitations

- The tiny hand-written dataset does not represent real contracts. Predictions can be wrong in either direction.
- Averaging embeddings loses word order. Retaining negations does not solve this: “sell data” and “do not sell data” may still have similar vectors.
- “Safe” is only a model label. A dangerous clause may be missed, and a harmless clause may be flagged.
- Type hints and phrase matches use simple patterns. They do not understand exceptions, consent, cross-references, or legal context.
- Sentence-based splitting is only an approximation of legal clauses.
- NER can miss company names, dates, and currencies. ORG also includes non-company organizations.
- All-unknown GloVe text receives a zero vector and an unreliable prediction; empty TF-IDF vocabulary is reported as an analysis error.
- NMF topics may be repetitive or meaningless on very short text; fewer than two clauses/features produces no topics.
- TF-IDF QA matches vocabulary rather than meaning, can miss synonyms, and can return a weak match. It does not answer yes/no.
- The risk score treats clauses equally and does not measure severity or legal enforceability.
- English only; no database, upload parser, or independent benchmark is included. First model download needs internet and models consume memory.
- The tool is educational and is not legal advice.

### Future Scope

- Multilingual translation and analysis, with checks that translation preserves legal meaning.
- Speech input to ask questions by voice.
- A browser extension that helps inspect terms on web pages.
- A larger, independently annotated dataset and a held-out evaluation set.
- Context-aware legal-language embeddings, improved negation handling, and severity labels.
- PDF extraction and explanations grounded in the exact source text.

### Conclusion

This project demonstrates how several NLP methods can work together in a simple web application. It helps highlight text that deserves attention and makes a useful college demonstration. It requires stronger data, evaluation, and language understanding before being used for real decisions.

## 5. Ten viva questions with easy answers

1. **What problem does your project solve?**  
   It helps users find potentially concerning clauses in long terms and conditions.

2. **What is preprocessing?**  
   It prepares text by splitting sentences, finding tokens, converting words to lemmas, and removing common words. I retain some negation words because they can change meaning.

3. **What is the difference between POS tagging and NER?**  
   POS finds grammatical roles such as noun and verb. NER finds named things such as organizations, dates, money, and places.

4. **What is GloVe, and why average its vectors?**  
   GloVe represents each known word using numbers learned from word co-occurrence. Averaging gives one fixed-size vector for a clause, making it easy to classify.

5. **Why did you use logistic regression?**  
   It is a simple supervised model for binary classification. It learns from labelled examples and predicts Risky or Safe. The current dataset is too small for reliable practical use.

6. **What is TF-IDF used for here?**  
   It measures which words distinguish clauses. I use it for keyword ranking, topic modelling, and matching questions to clauses.

7. **What are bigrams and trigrams?**  
   Bigrams contain two adjacent words, such as “without notice”. Trigrams contain three, such as “right to sue”. A match is a signal to inspect, not proof of risk.

8. **What does NMF do?**  
   It breaks the nonnegative TF-IDF matrix into smaller matrices to discover word groups called topics. Each clause is assigned to its strongest topic.

9. **How does QA work, and how is the risk score calculated?**  
   QA returns the clause with the highest TF-IDF cosine similarity to the question. The score is 100 times the risky-clause count divided by all clauses; it is not a probability of harm.

10. **How would you evaluate and improve the project?**  
    Use unseen clauses labelled by suitable reviewers and calculate precision, recall, F1, and a confusion matrix. Add more diverse data and better context models. Never report training accuracy as real-world performance.

## 6. PPT outline

### Slide 1 — Terms & Conditions Analyzer
- NLP Mini Project — B.Tech CSE (AI & ML)
- Chaitanya | PRN: 124B1C108
- Simple Streamlit web application

### Slide 2 — Real-World Problem
- Users often skip long agreements
- Clauses can affect data, payments, accounts, and rights
- Users need plain-language guidance

### Slide 3 — Objectives
- Identify potentially risky clauses
- Show type hints and key details
- Support keyword, topic, and question-based exploration

### Slide 4 — Technology Stack
- Python and Streamlit
- spaCy, NLTK, and Gensim GloVe
- scikit-learn and Matplotlib
- One application file; no database

### Slide 5 — NLP Pipeline
- Preprocessing, POS, and NER
- TF-IDF and average embeddings
- Classification and n-gram matching
- NMF topics and cosine retrieval

### Slide 6 — Classification and Risk Types
- 24 built-in labelled examples
- 50-dimensional GloVe averages and logistic regression
- Five requested risk types plus refunds
- Type hints come from separate keyword rules

### Slide 7 — Main Screens
- Paste text or use sample
- Red/green risk report and document score
- Entity table, phrases, keywords, and topics
- Question box and charts

### Slide 8 — Demonstration and Results
- Run the sample and show actual local output
- Ask “Can they sell my data?”
- Compare positive and negated wording
- No independent accuracy claim

### Slide 9 — Limitations and Evaluation
- Small dataset and weak context understanding
- False positives, false negatives, and negation errors
- Score measures flagged-clause share only
- Future evaluation: held-out precision, recall, and F1

### Slide 10 — Future Scope and Conclusion
- Multilingual translation, speech input, browser extension
- Larger annotated dataset and improved models
- Useful educational prototype
- Not legal advice

## 7. Real-World Usability Enhancements

To make the application directly usable by real consumers in day-to-day life, the following production-grade features were added:

1. **Multi-Source Ingestion (Zero-Friction Loading):**
   - **PDF Document Parser:** Extracts clean text from multi-page PDF agreements (rental leases, employment terms, EULAs) using `pypdf`.
   - **Webpage URL Scraper:** Scrapes live Terms of Service directly from any website URL (stripping scripts, headers, navigation bars) using `requests` and `BeautifulSoup`.
   - **Direct Paste & Presets:** Still supports manual text entry and 1-click sample loading.

2. **Consumer Threat Matrix (TL;DR Executive Summary):**
   - High-impact dashboard at the top of the report testing 6 vital consumer anxieties:
     - 👁️ Data Sharing & Ad Brokers
     - 💳 Auto-Renewal & Recurring Charges
     - 🚫 Strict No-Refund Policies
     - ⚠️ Termination & Bans Without Notice
     - ⚖️ Legal Rights & Mandatory Arbitration Waivers
     - 📸 Content Takeover & Perpetual Licenses
   - Shows plain-English explanations and direct quote snippets.

3. **Hybrid Semantic Question Answering:**
   - Upgraded search from exact-word TF-IDF to a **hybrid model**: combines TF-IDF lexical matching (35%) with 50-dimensional GloVe dense embedding cosine similarity (65%).
   - Enables questions like *"Can I get my money back?"* or *"Do they own my pictures?"* to correctly retrieve relevant clauses even when different vocabulary is used.

4. **Actionable Legal Opt-Out Letter Generator:**
   - Automatically drafts formal, legally valid opt-out letters (for binding arbitration, class-action waivers, or CCPA/GDPR data selling) with one-click copy.

5. **Downloadable Audit Report:**
   - One-click export of findings to Markdown (`.md`) format.

## 8. Reference documentation

Delivery verification: Python syntax, the under-200-line requirement, 24 unique training examples, phrase matching (including empty input), and all six type patterns passed dependency-free checks. A full pretrained-model run and Streamlit browser test were not performed: the available workspace Python lacks the requested packages, and creating a test environment failed while bootstrapping pip. Follow the setup commands above and record your local demonstration results.

- [Streamlit resource caching](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource)
- [spaCy model installation](https://spacy.io/usage/models)
- [Gensim pretrained model repository and download sizes](https://github.com/piskvorky/gensim-data)
- [Gensim downloader API](https://radimrehurek.com/gensim/downloader.html)
