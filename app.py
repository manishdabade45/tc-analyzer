import re
import datetime
import io
import numpy as np
import streamlit as st
import spacy
import spacy.cli
import gensim.downloader as api
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import pypdf
from nltk import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

# --- Tiny educational dataset: first 12 risky (1), next 12 safe (0) ---
TRAIN = [
    'We may sell your personal data to advertisers.',
    'We share your information with third parties for marketing.',
    'Your subscription renews automatically and your card is charged.',
    'Payments are non refundable and no refunds are provided.',
    'We may terminate your account without notice or reason.',
    'We can suspend access at any time without warning.',
    'You waive your right to sue us in court.',
    'We accept no liability for any loss or damage.',
    'You grant us ownership of all uploaded content.',
    'We receive a perpetual irrevocable license to your content.',
    'We may change fees without notice and charge your account.',
    'You must resolve disputes through binding arbitration.',
    'We do not sell your personal data to advertisers.',
    'We share information only with your explicit consent.',
    'Your subscription ends unless you choose to renew it.',
    'You may request a full refund within thirty days.',
    'We provide notice and an appeal before closing an account.',
    'You can cancel your subscription at any time.',
    'You retain your right to bring a claim in court.',
    'We remain responsible for damage caused by our negligence.',
    'You retain ownership of all uploaded content.',
    'Your content license ends when you delete the content.',
    'We notify you before any price change takes effect.',
    'You may download and delete your personal information.'
]

SAMPLE = ('Acme Ltd operates in London. These terms start on January 1, 2027.\n'
          'We may sell your personal data to advertisers.\n'
          'Your subscription renews automatically at $20 per month.\n'
          'We may terminate your account without notice or reason.\n'
          'You waive your right to sue us in court.\n'
          'You grant us ownership of all uploaded content.\n'
          'Payments are non refundable and no refunds are provided.\n'
          'You may download and delete your personal information.')

RULES = {
    'Data sharing': (
        r'\b(data|information|advertis\w*|third parties|sell|sharing)\b',
        'Your personal information or browsing data may be shared with or sold to third-party advertisers.',
        '👁️ Data & Privacy'
    ),
    'Auto-renewal': (
        r'\b(renew\w*|subscription|recurring|automatic billing|billed)\b',
        'Your payment method may be charged repeatedly and automatically without individual prior confirmation.',
        '💳 Auto-Renewal'
    ),
    'Unfair termination': (
        r'\b(terminat\w*|suspend\w*|closing|without notice|at any time|sole discretion)\b',
        'The provider can close, ban, or suspend your account at any time without advance warning or reason.',
        '⚠️ Account Termination'
    ),
    'Liability waiver': (
        r'\b(waiv\w*|liability|sue|arbitration|dispute\w*|class action|indemnif\w*)\b',
        'You give up constitutional rights to bring claims in court, participate in class actions, or seek damages.',
        '⚖️ Legal Rights & Arbitration'
    ),
    'Content ownership': (
        r'\b(content|ownership|license|uploaded|perpetual|irrevocable|royalty-free)\b',
        'The company claims broad, perpetual, or irrevocable commercial rights over any materials you submit.',
        '📸 Content Rights'
    ),
    'No refund': (
        r'\b(refund\w*|non refundable|final sale|no return)\b',
        'All purchases and payments may be completely non-refundable regardless of user satisfaction.',
        '🚫 Refund Policy'
    )
}

PHRASES = {'without notice', 'third parties', 'no refunds', 'non refundable',
           'renews automatically', 'binding arbitration', 'right to sue', 'irrevocable license'}

# --- Caching Large NLP Resources ---
@st.cache_resource
def load_resources():
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        spacy.cli.download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
    glove = api.load('glove-wiki-gigaword-50')
    return nlp, glove

# --- Document Ingestion Helpers ---
def extract_text_from_pdf(uploaded_file):
    reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
    text = '\n'.join([page.extract_text() or '' for page in reader.pages])
    return text.strip(), len(reader.pages)

def extract_text_from_url(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    resp = requests.get(url, headers=headers, timeout=12)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript', 'svg']):
        tag.decompose()
    paragraphs = [p.get_text().strip() for p in soup.find_all(['p', 'li', 'h1', 'h2', 'h3']) if len(p.get_text().strip()) > 20]
    return '\n'.join(paragraphs).strip()

# --- NLP Processing Pipeline ---
def preprocess(text, nlp):
    docs = [nlp(part.strip()) for part in re.split(r'[\n;]+', text) if part.strip()]
    clauses = [s.text.strip() for doc in docs for s in doc.sents if s.text.strip()]
    parsed = list(nlp.pipe(clauses))
    tokens = [[t.lemma_.lower() for t in d if t.is_alpha and
               (not t.is_stop or t.lower_ in {'no', 'not', 'never', 'without'})] for d in parsed]
    return clauses, parsed, tokens

def pos_tagging(docs):
    return [{'Word': t.text, 'POS': t.pos_} for d in docs for t in d if t.is_alpha]

def named_entities(docs):
    labels = {'ORG': 'Company / organization', 'MONEY': 'Money', 'DATE': 'Date',
              'GPE': 'Place', 'LOC': 'Place'}
    return [{'Entity': e.text, 'Type': labels[e.label_]} for d in docs
            for e in d.ents if e.label_ in labels]

def tfidf(tokens):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([' '.join(t) for t in tokens])
    weights = np.asarray(matrix.mean(axis=0)).ravel()
    ids = weights.argsort()[::-1][:10]
    return vectorizer, matrix, [(vectorizer.get_feature_names_out()[i], float(weights[i])) for i in ids]

def embeddings(tokens, glove):
    return np.array([np.mean([glove[w] for w in words if w in glove], axis=0)
                     if any(w in glove for w in words) else np.zeros(50) for words in tokens])

@st.cache_resource
def train_classifier(_nlp, _glove):
    _, _, tokens = preprocess('\n'.join(TRAIN), _nlp)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000, random_state=42))
    return model.fit(embeddings(tokens, _glove), [1] * 12 + [0] * 12)

def classify(tokens, glove, model):
    return model.predict(embeddings(tokens, glove)).astype(bool)

def risky_phrases(clause):
    words = re.findall(r'[a-z]+', clause.lower())
    return sorted({' '.join(g) for n in (2, 3) for g in ngrams(words, n)} & PHRASES)

def topic_model(matrix, vectorizer):
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return []
    nmf = NMF(n_components=min(3, *matrix.shape), init='random', random_state=42, max_iter=500)
    membership = nmf.fit_transform(matrix)
    names = vectorizer.get_feature_names_out()
    return [{'Topic': i + 1, 'Words': ', '.join(names[row.argsort()[-5:][::-1]]),
             'Clauses': ', '.join(str(j + 1) for j in range(len(membership)) if membership[j].argmax() == i)}
            for i, row in enumerate(nmf.components_)]

def answer_question(question, nlp, vectorizer, matrix, clauses, glove, clause_embeds):
    _, _, tokens = preprocess(question, nlp)
    q_str = ' '.join(w for sentence in tokens for w in sentence)
    if not q_str.strip():
        return None, 0.0, 0.0, 0.0
    
    # 1. Lexical TF-IDF match
    query_vec = vectorizer.transform([q_str])
    tfidf_sims = cosine_similarity(query_vec, matrix).ravel()
    
    # 2. Semantic GloVe embedding match
    q_embed = embeddings(tokens, glove)
    q_norm = np.linalg.norm(q_embed)
    if q_norm > 0:
        c_norms = np.linalg.norm(clause_embeds, axis=1, keepdims=True)
        c_norms[c_norms == 0] = 1e-9
        normed_clauses = clause_embeds / c_norms
        glove_sims = np.clip(np.dot(normed_clauses, (q_embed.flatten() / q_norm)), 0, 1)
    else:
        glove_sims = np.zeros(len(clauses))
        
    # Hybrid score combining meaning and exact keywords
    hybrid_scores = 0.35 * tfidf_sims + 0.65 * glove_sims
    best_idx = int(hybrid_scores.argmax())
    best_score = float(hybrid_scores[best_idx])
    return (clauses[best_idx], best_score, float(tfidf_sims[best_idx]), float(glove_sims[best_idx])) if best_score > 0.15 else (None, 0.0, 0.0, 0.0)

def risk_types(clause):
    return [name for name, (pattern, _, _) in RULES.items() if re.search(pattern, clause.lower())]

# --- Streamlit Application UI ---
st.set_page_config(page_title='Terms & Conditions Analyzer — Real-Life Edition', layout='wide', page_icon='🛡️')

st.title('🛡️ Terms & Conditions Analyzer')

# Sidebar for zero-friction document ingestion
with st.sidebar:
    st.header('📥 Input Document')
    input_mode = st.radio('Choose how to load terms:', ['✍️ Paste Text', '🌐 Fetch from Web URL', '📄 Upload PDF Document'])
    
    if input_mode == '✍️ Paste Text':
        if st.button('📋 Load Sample T&C'):
            st.session_state['terms_area'] = SAMPLE
            st.rerun()
    elif input_mode == '🌐 Fetch from Web URL':
        url_input = st.text_input('Website Terms URL', placeholder='https://example.com/terms')
        if st.button('Fetch & Scrape Terms') and url_input.strip():
            with st.spinner('Scraping clean text from webpage...'):
                try:
                    loaded_text = extract_text_from_url(url_input.strip())
                    if not loaded_text:
                        st.warning('No suitable text content found at this URL. The site might require login or JavaScript.')
                    else:
                        st.session_state['terms_area'] = loaded_text[:30000]
                        st.rerun()
                except Exception as e:
                    st.error(f'Failed to fetch URL: {e}')
    elif input_mode == '📄 Upload PDF Document':
        uploaded_pdf = st.file_uploader('Upload contract or terms PDF', type=['pdf'])
        if uploaded_pdf is not None:
            if st.session_state.get('last_uploaded_pdf') != uploaded_pdf.name:
                with st.spinner('Extracting text from PDF pages...'):
                    try:
                        pdf_text, num_pages = extract_text_from_pdf(uploaded_pdf)
                        if not pdf_text:
                            st.warning('Could not extract text from this PDF (it might be scanned images).')
                        else:
                            st.session_state['terms_area'] = pdf_text[:30000]
                            st.session_state['last_uploaded_pdf'] = uploaded_pdf.name
                            st.rerun()
                    except Exception as e:
                        st.error(f'Error reading PDF: {e}')

    st.markdown('---')
    st.info('🔒 **100% Local & Private:** All NLP analysis runs completely on your machine. No contract data is sent to external servers.')

# Main Text Area
if 'terms_area' not in st.session_state:
    st.session_state['terms_area'] = ''

text = st.text_area('Terms & Conditions Text (up to 30,000 characters):', height=180, max_chars=30000, key='terms_area')

if not text.strip():
    st.info('👉 Select an input method in the sidebar (paste text, fetch a URL, or upload a PDF) to start analysis.')
    st.stop()

# Execution of NLP Pipeline
try:
    with st.spinner('Analyzing legal clauses with spaCy, GloVe & Logistic Regression...'):
        nlp, glove = load_resources()
        clauses, docs, tokens = preprocess(text, nlp)
        if not clauses:
            st.warning('No clear clauses could be extracted. Please verify the input text.')
            st.stop()
        vectorizer, matrix, keywords = tfidf(tokens)
        clause_embeds = embeddings(tokens, glove)
        clf = train_classifier(nlp, glove)
        risky = classify(tokens, glove, clf)
except Exception as error:
    st.error(f'Analysis could not finish: {error}')
    st.info('Verify input text and internet connection for model caching.'); st.stop()

count = int(risky.sum())
risk_pct = 100 * count / len(clauses)

# Navigation Tabs
tab_summary, tab_report, tab_details, tab_insights, tab_qa, tab_optout = st.tabs([
    '⚡ TL;DR Executive Gotchas',
    '📋 Clause Risk Report',
    '🔍 Key Details & POS',
    '📊 NLP Insights & Charts',
    '💬 Semantic Q&A',
    '✉️ Opt-Out Letter Generator'
])

# --- Tab 1: Executive Gotchas Summary ---
with tab_summary:
    st.subheader('Consumer Threat Matrix: At a Glance')
    col1, col2, col3 = st.columns(3)
    
    # Calculate category triggers across all clauses
    cat_matches = {}
    for cat_name, (pattern, expl, icon_title) in RULES.items():
        matched_clauses = [c for c in clauses if re.search(pattern, c.lower())]
        cat_matches[cat_name] = matched_clauses
        
    cols = [col1, col2, col3, col1, col2, col3]
    for idx, (cat_name, (pattern, expl, icon_title)) in enumerate(RULES.items()):
        with cols[idx]:
            found = len(cat_matches[cat_name]) > 0
            badge_color = '🔴' if found else '🟢'
            st.markdown(f"### {badge_color} {icon_title}")
            if found:
                st.markdown(f"**Found in {len(cat_matches[cat_name])} clause(s)**")
                st.caption(expl)
                with st.expander('Show matched clause'):
                    st.write(cat_matches[cat_name][0])
            else:
                st.caption('No explicit high-risk pattern detected in this category.')
            st.markdown('---')
            
    # Risk Score Banner
    st.metric('Overall Risk Rating', f'{risk_pct:.0f}/100', delta=f'{count} risky clauses flagged', delta_color='inverse')
    
    # Export Report
    report_md = f"# Terms & Conditions Analysis Report\nGenerated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    report_md += f"**Overall Risk Score:** {risk_pct:.0f}/100 ({count} risky clauses of {len(clauses)})\n\n"
    report_md += "## Identified Consumer Traps:\n"
    for cat_name, items in cat_matches.items():
        if items:
            report_md += f"- **{cat_name}:** {len(items)} clause(s) found. Example: \"{items[0]}\"\n"
    st.download_button('📥 Download Analysis Report (.md)', data=report_md, file_name='terms_analysis_report.md', mime='text/markdown')

# --- Tab 2: Detailed Clause Report ---
with tab_report:
    st.subheader('Detailed Clause-by-Clause Evaluation')
    st.write(f'Evaluated {len(clauses)} clauses. Found {count} clauses flagged as potentially risky:')
    for i, (clause, flag) in enumerate(zip(clauses, risky), 1):
        types = risk_types(clause) if flag else []
        (st.error if flag else st.success)(f'{i}. {"Risky" if flag else "Safe"}: {clause}')
        if flag:
            st.write('**Type hints:** ' + ', '.join(types or ['Other / general liability risk']))
            explanations = [RULES[t][1] for t in types if t in RULES]
            if explanations:
                st.caption('💡 ' + ' '.join(explanations))

# --- Tab 3: Key Details & POS ---
with tab_details:
    st.subheader('Named Entities Identified')
    st.dataframe(named_entities(docs) or [{'Entity': 'None detected', 'Type': '-'}], use_container_width=True)
    with st.expander('Grammatical POS Tags (First 200 words)'):
        st.dataframe(pos_tagging(docs)[:200], use_container_width=True)

# --- Tab 4: NLP Insights & Charts ---
with tab_insights:
    st.subheader('Risky Phrase Matches (Multi-Word N-Grams)')
    detected_phrases = [{'Clause': i + 1, 'Phrases': ', '.join(risky_phrases(c))} for i, c in enumerate(clauses) if risky_phrases(c)]
    st.dataframe(detected_phrases or [{'Clause': '-', 'Phrases': 'No listed high-risk n-grams detected'}], use_container_width=True)
    
    col_k, col_t = st.columns(2)
    with col_k:
        st.subheader('Distinguishing TF-IDF Keywords')
        st.dataframe([{'Keyword': w, 'TF-IDF Weight': round(v, 4)} for w, v in keywords], use_container_width=True)
    with col_t:
        st.subheader('NMF Topic Clusters')
        st.dataframe(topic_model(matrix, vectorizer), use_container_width=True)
        
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(['Risky', 'Safe'], [count, len(clauses) - count], color=['#e63946', '#2a9d8f'])
    axes[0].set_ylabel('Number of Clauses')
    axes[0].set_title('Clause Safety Distribution')
    axes[1].barh([w for w, _ in keywords][::-1], [v for _, v in keywords][::-1], color='#457b9d')
    axes[1].set_title('Mean TF-IDF Feature Weights')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# --- Tab 5: Semantic Q&A ---
with tab_qa:
    st.subheader('Ask Questions in Plain English')
    st.caption('Combines TF-IDF keyword lookup with 50-dimensional GloVe semantic vector matching to retrieve answers even if exact words differ.')
    
    example_col1, example_col2, example_col3 = st.columns(3)
    preset_q = None
    if example_col1.button('Can they sell my data?'):
        preset_q = 'Can they sell my data?'
    if example_col2.button('Can I get a refund?'):
        preset_q = 'Can I get a refund?'
    if example_col3.button('Do they own my pictures?'):
        preset_q = 'Do they own my pictures?'
        
    question = st.text_input('Your Question:', value=preset_q or '', placeholder='e.g., Will they charge my card automatically?')
    if question.strip():
        answer, hybrid_s, tfidf_s, glove_s = answer_question(question, nlp, vectorizer, matrix, clauses, glove, clause_embeds)
        if answer:
            st.info(f'**Closest Matching Clause:**\n\n> "{answer}"')
            st.caption(f'📊 Match Confidence: Hybrid: {hybrid_s:.2f} | Semantic (GloVe): {glove_s:.2f} | Lexical (TF-IDF): {tfidf_s:.2f}')
        else:
            st.warning('No sufficiently matching clause found in these terms. Try rewording your question.')

# --- Tab 6: Actionable Opt-Out Letter Generator ---
with tab_optout:
    st.subheader('✉️ Generate Legal Opt-Out Notice')
    st.write('Many consumer agreements legally permit you to opt out of **binding arbitration**, **class action waivers**, or **data selling** if sent within 30 days of opening an account.')
    
    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        user_name = st.text_input('Your Full Name', placeholder='e.g., Jane Doe')
        account_email = st.text_input('Your Account Email', value='user@example.com')
    with opt_col2:
        company_name = st.text_input('Service / Company Name', value='Acme Corporation')
        opt_type = st.selectbox('Opt-Out Subject', [
            'Binding Arbitration & Class Action Waiver',
            'Sale and Sharing of Personal Data (CCPA/GDPR)',
            'Both Arbitration and Data Sale'
        ])
        
    today_str = datetime.date.today().strftime('%B %d, %Y')
    sender_name = user_name.strip() if user_name else '[Your Full Name]'
    letter_text = f"""SUBJECT: Formal Written Opt-Out Notice — {opt_type}

Date: {today_str}
To: Legal & Privacy Department, {company_name}

Dear Legal and Privacy Compliance Team,

Please accept this written notice that I, {sender_name}, hereby exercise my legal right to formally OPT OUT of the following provisions in your Terms of Service / User Agreement:
- {opt_type}

User Account Details:
- Full Name: {sender_name}
- Registered Email Address: {account_email}
- Notice Date: {today_str}

Please record this opt-out in your active records and provide written confirmation to my registered email address that this request has been processed.

Sincerely,
{sender_name}
"""
    st.text_area('Copy and send this notice to the company\'s legal or privacy email:', value=letter_text, height=260)
    st.success('💡 Tip: Send this within 30 days of registration to legal@ or privacy@ the provider\'s domain.')
