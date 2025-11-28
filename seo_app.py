import streamlit as st
import pandas as pd
import requests
import json
import io
import csv
import glob
import urllib.parse
from bs4 import BeautifulSoup
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# --- CONFIGURATION ---
st.set_page_config(page_title="AI SEO Auditor", page_icon="🧠", layout="wide")

if 'seo_results' not in st.session_state:
    st.session_state['seo_results'] = []

# --- AUTHENTICATION ---
def get_creds():
    creds_info = None
    if "gcp_service_account" in st.secrets:
        try:
            creds_info = dict(st.secrets["gcp_service_account"])
            if "private_key" in creds_info:
                creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
        except: pass
    
    if not creds_info:
        for k in glob.glob("*.json"):
            if "service_account" in k or "qc" in k:
                try:
                    with open(k, "r") as f:
                        creds_info = json.load(f)
                        break
                except: continue

    if creds_info:
        return service_account.Credentials.from_service_account_info(
            creds_info, 
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    return None

# --- AI ANALYSIS (Anti-Hallucination Prompt) ---
def analyze_with_gemini(content_text, meta_data, schema_data, creds):
    try:
        vertexai.init(project=creds.project_id, location="us-central1", credentials=creds)
        model = GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""
        Act as a Strict Technical SEO Validator.
        
        CONTEXT:
        We are auditing a medical website. Accuracy is critical.
        
        1. PAGE CONTENT (Summary): 
        "{content_text[:2000]}"

        2. CURRENT METADATA: 
        Title: {meta_data['Title']}
        Desc: {meta_data['Meta Description']}

        3. CURRENT SCHEMA (@type found): 
        {schema_data}

        TASK:
        1. Rating: Rate the Meta Title/Content alignment (High/Medium/Low).
        2. Missing Schema: Identify 1 specific Schema.org type missing based on content. 
           Rule: ONLY suggest official types from Schema.org (e.g. MedicalWebPage, FAQPage, Physician). Do NOT invent types.
        3. Critique: Write 1 short sentence improving the Meta Description.

        OUTPUT JSON ONLY: {{ "rating": "...", "schema_suggestion": "...", "meta_critique": "..." }}
        """
        
        response = model.generate_content(prompt, generation_config=GenerationConfig(response_mime_type="application/json"))
        return json.loads(response.text)
    except Exception as e:
        return {"rating": "Error", "schema_suggestion": str(e), "meta_critique": ""}

# --- SCRAPER ---
def scrape_seo_data(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (SEO-Auditor)'}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # 1. Basic Meta
        title = soup.find('title').get_text().strip() if soup.find('title') else "MISSING"
        meta = soup.find('meta', attrs={'name': 'description'})
        meta_desc = meta['content'].strip() if meta else "MISSING"
        
        # 2. Schema Extraction & JSON Validation
        schemas = []
        valid_json = True
        for s in soup.find_all('script', type='application/ld+json'):
            if s.string:
                try:
                    # Verify it is actual JSON (Python is the Judge here, not AI)
                    json.loads(s.string) 
                    schemas.append(s.string)
                except json.JSONDecodeError:
                    valid_json = False
        
        # 3. Content Extraction
        content_area = soup.find(class_="page-content-area")
        if content_area:
            body_text = content_area.get_text(separator=' ').strip()
        else:
            for tag in soup(["script", "style", "nav", "footer"]): tag.decompose()
            body_text = soup.get_text(separator=' ').strip()

        return {
            "Title": title,
            "Meta Description": meta_desc,
            "Schema Raw": schemas,
            "JSON Valid": valid_json, # Python judgment
            "Body Text": body_text
        }
    except Exception as e:
        return {"Error": str(e)}

# --- UI ---
st.title("🧠 AI-Powered SEO Auditor")
st.markdown("Combines **Python Logic** (Fact Check) with **Gemini AI** (Strategy).")

# Sidebar
with st.sidebar:
    st.header("Settings")
    use_ai = st.checkbox("Enable AI Analysis", value=True)
    use_staging = st.checkbox("Override Domain")
    staging_domain = st.text_input("Staging Domain") if use_staging else ""

creds = get_creds()
if not creds:
    st.error("⚠️ Google Cloud Credentials not found.")
    st.stop()

csv_file = st.file_uploader("Upload Sitemap CSV", type="csv")

if st.button("Run Audit", type="primary"):
    if csv_file:
        stringio = io.StringIO(csv_file.getvalue().decode("utf-8-sig"))
        rows = list(csv.DictReader(stringio))
        
        results = []
        bar = st.progress(0)
        status = st.empty()
        
        for i, row in enumerate(rows):
            page_title = row.get('Page Title', 'Unknown')
            url = row.get('URL', '')
            
            if use_staging and staging_domain:
                from urllib.parse import urlparse
                path = urlparse(url).path
                url = f"https://{staging_domain}{path}"
            
            status.text(f"Analyzing: {page_title}...")
            
            # 1. Scrape
            data = scrape_seo_data(url)
            
            if "Error" in data:
                results.append({"Page Title": page_title, "Status": "ERROR", "Error": data['Error']})
            else:
                # 2. Flatten Schema Types
                schema_list = []
                for s in data['Schema Raw']:
                    try:
                        j = json.loads(s)
                        if '@graph' in j:
                            for item in j['@graph']: schema_list.append(item.get('@type', 'Unknown'))
                        else:
                            schema_list.append(j.get('@type', 'Unknown'))
                    except: pass
                
                flat_schema = []
                for item in schema_list:
                    if isinstance(item, list): flat_schema.extend(item)
                    else: flat_schema.append(item)
                
                # 3. AI Analysis
                ai_feedback = {}
                if use_ai:
                    ai_feedback = analyze_with_gemini(
                        data['Body Text'], 
                        {"Title": data['Title'], "Meta Description": data['Meta Description']},
                        flat_schema,
                        creds
                    )

                # 4. Generate Google Validator Link
                google_test_url = f"https://search.google.com/test/rich-results?url={urllib.parse.quote(url)}"

                results.append({
                    "Page Title": page_title,
                    "Schema Syntax": "✅ Valid" if data['JSON Valid'] else "❌ Syntax Error",
                    "Schema Types": ", ".join(set(flat_schema)),
                    "AI Suggestion": ai_feedback.get('schema_suggestion', '-'),
                    "AI Alignment": ai_feedback.get('rating', '-'),
                    "AI Critique": ai_feedback.get('meta_critique', '-'),
                    "Verify": google_test_url # Link for human verification
                })
            
            bar.progress((i+1)/len(rows))
        
        status.text("Audit Complete!")
        st.session_state['seo_results'] = results

if st.session_state['seo_results']:
    df = pd.DataFrame(st.session_state['seo_results'])
    
    def color_ai(val):
        if val == "High": return 'color: green; font-weight: bold'
        if val == "Low": return 'color: red; font-weight: bold'
        if val == "❌ Syntax Error": return 'color: red; font-weight: bold'
        return ''

    # Make the Verify link clickable in the table
    st.dataframe(
        df.style.applymap(color_ai, subset=['AI Alignment', 'Schema Syntax']), 
        column_config={"Verify": st.column_config.LinkColumn("Google Validator")},
        use_container_width=True
    )
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Report", csv, "ai_seo_audit.csv", "text/csv")