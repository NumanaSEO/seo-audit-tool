import streamlit as st
import pandas as pd
import requests
import json
import io
import csv
import glob
import urllib.parse
import difflib
import time
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

# --- AI ANALYSIS (Now includes NLP Check) ---
def analyze_with_gemini(content_text, meta_data, schema_data, creds):
    try:
        vertexai.init(project=creds.project_id, location="us-central1", credentials=creds)
        model = GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""
        Act as a Strict SEO Auditor.
        
        1. PAGE CONTENT SAMPLE: "{content_text[:2500]}"
        2. METADATA: 
           - Title: {meta_data['Title']}
           - Desc: {meta_data['Meta Description']}
        3. SCHEMA FOUND: {schema_data}

        YOUR ANALYSIS TASKS:
        
        1. LOCAL SEO CHECK (Critical):
           - Is this a location page (address/map in content)? If YES, is 'MedicalClinic'/'LocalBusiness' schema present?
           - If Location Page AND Missing Schema -> Rate "Low".

        2. GOOGLE NLP CHECK (Snippet Retention):
           - Does the Meta Description explicitly mention entities found in the H1/Body?
           - If vague ("We offer services") -> Risk: "Likely Rewrite".
           - If specific ("We offer CBT and ADHD testing") -> Risk: "Likely Keep".

        3. RATING (High/Medium/Low): 
           - Rate Title/Content alignment. Penalize generic titles.

        4. WRITING QUALITY: 
           - Grade the Meta Description (Professional/Awkward/Poor).

        5. SCHEMA GAP: 
           - Suggest 1 specific Schema.org type missing (Official types only).

        6. CRITIQUE: 
           - Write 1 sentence on how to fix the tags.

        OUTPUT JSON ONLY: {{ 
            "rating": "...", 
            "writing_quality": "...", 
            "google_rewrite_risk": "Likely Keep/Likely Rewrite",
            "schema_suggestion": "...", 
            "meta_critique": "..." 
        }}
        """
        
        response = model.generate_content(prompt, generation_config=GenerationConfig(response_mime_type="application/json"))
        return json.loads(response.text)
    except Exception as e:
        return {"rating": "Error", "writing_quality": "Error", "google_rewrite_risk": "Error", "schema_suggestion": str(e), "meta_critique": ""}

# --- SCORING ---
def calculate_score(data, ai_result):
    score = 100
    reasons = []

    # Technical Checks
    if not data['JSON Valid']:
        score -= 30
        reasons.append("Broken Schema Syntax (-30)")
    if data['Title'] == "MISSING":
        score -= 20
        reasons.append("Missing Title (-20)")
    if data['Meta Description'] == "MISSING":
        score -= 20
        reasons.append("Missing Meta Desc (-20)")

    # Echo/Auto-Gen Check
    if data['Echo Score'] > 85:
        score -= 15
        reasons.append("⚠️ Auto-Generated Desc (-15)")

    # Length Checks
    t_len = len(data['Title'])
    if t_len < 10 or t_len > 70:
        score -= 5
        reasons.append(f"Bad Title Length ({t_len}) (-5)")

    # AI Quality (Includes NLP & Local Checks)
    if ai_result.get('rating') == "Low":
        score -= 25
        reasons.append("Low Relevance/Missing Local Schema (-25)")
    
    if ai_result.get('google_rewrite_risk') == "Likely Rewrite":
        score -= 10
        reasons.append("Vague Desc (Google will rewrite) (-10)")

    if ai_result.get('writing_quality') == "Poor":
        score -= 15
        reasons.append("Poor Grammar (-15)")

    return max(0, score), ", ".join(reasons)

# --- SCRAPER ---
def scrape_seo_data(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (SEO-Auditor)'}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # 1. Metadata
        title = soup.find('title').get_text().strip() if soup.find('title') else "MISSING"
        meta = soup.find('meta', attrs={'name': 'description'})
        meta_desc = meta['content'].strip() if meta else "MISSING"
        
        # 2. Schema
        schemas = []
        valid_json = True
        for s in soup.find_all('script', type='application/ld+json'):
            if s.string:
                try:
                    json.loads(s.string)
                    schemas.append(s.string)
                except json.JSONDecodeError:
                    valid_json = False
        
        # 3. Content
        content_area = soup.find(class_="page-content-area")
        if content_area:
            body_text = content_area.get_text(separator=' ').strip()
        else:
            for tag in soup(["script", "style", "nav", "footer"]): tag.decompose()
            body_text = soup.get_text(separator=' ').strip()

        # 4. Echo Check
        echo_score = 0
        if meta_desc != "MISSING" and body_text:
            matcher = difflib.SequenceMatcher(None, meta_desc, body_text[:len(meta_desc) + 50])
            echo_score = matcher.ratio() * 100

        return {
            "Title": title,
            "Meta Description": meta_desc,
            "Schema Raw": schemas,
            "JSON Valid": valid_json,
            "Body Text": body_text,
            "Echo Score": echo_score
        }
    except Exception as e:
        return {"Error": str(e)}

# --- UI ---
st.title("🧠 AI-Powered SEO Auditor")

with st.sidebar:
    st.header("Settings")
    use_ai = st.checkbox("Enable AI Analysis", value=True)
    use_staging = st.checkbox("Override Domain")
    staging_domain = st.text_input("Staging Domain") if use_staging else ""

creds = get_creds()
if not creds:
    st.error("⚠️ Credentials missing. Please add secrets.toml or service_account.json.")
    st.stop()

csv_file = st.file_uploader("Upload Sitemap CSV", type="csv")

if st.button("Run Audit", type="primary"):
    if csv_file:
        stringio = io.StringIO(csv_file.getvalue().decode("utf-8-sig"))
        rows = list(csv.DictReader(stringio))
        
        results = []
        bar = st.progress(0)
        status = st.empty()
        total_rows = len(rows)
        
        for i, row in enumerate(rows):
            csv_title = row.get('Page Title')
            url = row.get('URL', '')
            
            if not url or str(url).strip() == "": continue

            if use_staging and staging_domain:
                from urllib.parse import urlparse
                path = urlparse(url).path
                url = f"https://{staging_domain}{path}"
            
            display_name = csv_title if csv_title and csv_title.strip() else url
            status.text(f"[{i+1}/{total_rows}] 🕷️ Scraping: {display_name}...")
            
            time.sleep(0.5) 
            data = scrape_seo_data(url)
            
            if "Error" in data:
                results.append({"Page Title": display_name, "URL": url, "Score": 0, "Status": "ERROR", "Error": data['Error']})
            else:
                if not csv_title or not csv_title.strip(): display_name = data['Title']

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
                
                ai_feedback = {}
                if use_ai:
                    status.text(f"[{i+1}/{total_rows}] 🤖 Analyzing: {display_name}...")
                    ai_feedback = analyze_with_gemini(
                        data['Body Text'], 
                        {"Title": data['Title'], "Meta Description": data['Meta Description']},
                        flat_schema,
                        creds
                    )

                final_score, score_log = calculate_score(data, ai_feedback)
                google_test_url = f"https://search.google.com/test/rich-results?url={urllib.parse.quote(url)}"
                gen_status = "🤖 Auto-Gen" if data['Echo Score'] > 85 else "✍️ Unique"

                results.append({
                    "Page Title": display_name,
                    "URL": url,
                    "Score": final_score,
                    "Score Log": score_log,
                    "Current Title": data['Title'],
                    "Len (T)": len(data['Title']),
                    "Current Desc": data['Meta Description'],
                    "Len (D)": len(data['Meta Description']),
                    "AI Rating": ai_feedback.get('rating', '-'),
                    "Google NLP Risk": ai_feedback.get('google_rewrite_risk', '-'),
                    "Writing Quality": ai_feedback.get('writing_quality', '-'),
                    "Source": gen_status,
                    "AI Critique": ai_feedback.get('meta_critique', '-'),
                    "AI Suggestion": ai_feedback.get('schema_suggestion', '-'),
                    "Schema": ", ".join(set(flat_schema)),
                    "Verify": google_test_url
                })
            
            bar.progress((i+1)/total_rows)
        
        status.success("Audit Complete!")
        results.sort(key=lambda x: x['Score'])
        st.session_state['seo_results'] = results

if st.session_state['seo_results']:
    df = pd.DataFrame(st.session_state['seo_results'])
    
    def color_rows(val):
        if val == "High" or val == "Likely Keep": return 'color: green; font-weight: bold'
        if val == "Low" or val == "Likely Rewrite": return 'color: red; font-weight: bold'
        if isinstance(val, int) and (val > 160 or val < 10): return 'color: orange; font-weight: bold'
        return ''

    def color_score(val):
        if isinstance(val, int):
            if val >= 90: return 'background-color: #d4edda; color: black; font-weight: bold' 
            if val >= 70: return 'background-color: #fff3cd; color: black; font-weight: bold' 
            return 'background-color: #f8d7da; color: black; font-weight: bold' 
        return ''

    st.dataframe(
        df.style.applymap(color_rows).applymap(color_score, subset=['Score']), 
        column_config={
            "Verify": st.column_config.LinkColumn("Google Validator"),
            "URL": st.column_config.LinkColumn("Live Page"),
            "Score": st.column_config.ProgressColumn("Health Score", format="%d", min_value=0, max_value=100),
        },
        use_container_width=True
    )
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Report", csv, "ai_seo_audit.csv", "text/csv")
