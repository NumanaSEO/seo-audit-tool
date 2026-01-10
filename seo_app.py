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

if "seo_results" not in st.session_state:
    st.session_state["seo_results"] = []

# --- AUTHENTICATION ---
def get_creds():
    creds_info = None
    if "gcp_service_account" in st.secrets:
        try:
            creds_info = dict(st.secrets["gcp_service_account"])
            if "private_key" in creds_info:
                creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
        except Exception:
            pass

    if not creds_info:
        for k in glob.glob("*.json"):
            if "service_account" in k or "qc" in k:
                try:
                    with open(k, "r") as f:
                        creds_info = json.load(f)
                        break
                except Exception:
                    continue

    if creds_info:
        return service_account.Credentials.from_service_account_info(
            creds_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
    return None


# --- CSV HELPERS (NEW) ---
def detect_csv_columns(rows):
    """Return (url_key, title_key, headers)."""
    if not rows:
        return None, None, []

    headers = [h for h in rows[0].keys() if h]
    headers_norm = {h.strip().lower(): h for h in headers}

    url_candidates = ["url", "page url", "page_url", "loc", "link", "address"]
    title_candidates = ["page title", "title", "page", "name"]

    url_key = None
    for cand in url_candidates:
        if cand in headers_norm:
            url_key = headers_norm[cand]
            break

    title_key = None
    for cand in title_candidates:
        if cand in headers_norm:
            title_key = headers_norm[cand]
            break

    # If there is only one column, assume it's the URL column
    if not url_key and len(headers) == 1:
        url_key = headers[0]

    return url_key, title_key, headers


# --- AI ANALYSIS (Now Prescribes Schema) ---
def analyze_with_gemini(content_text, meta_data, schema_data, creds):
    try:
        vertexai.init(project=creds.project_id, location="us-central1", credentials=creds)
        model = GenerativeModel("gemini-2.5-flash")

        prompt = f"""
        Act as a Technical SEO Expert.

        1. PAGE CONTENT: "{content_text[:2500]}"
        2. METADATA: Title: "{meta_data['Title']}" | Desc: "{meta_data['Meta Description']}"
        3. EXISTING SCHEMA: {schema_data}

        TASKS:
        1. LOCAL CHECK: If this is a physical location page, is 'MedicalClinic' present?
        2. RATING: Rate Title/Content alignment (High/Medium/Low).
        3. WRITING: Grade Desc quality (Professional/Awkward/Poor).

        4. **THE FIX (Meta Desc):**
           If the current Desc is 'Likely Rewrite' or 'Awkward', write a BETTER one (Max 160 chars).
           If it's good, return "Keep Current".

        5. **THE PRESCRIPTION (Schema):**
           Look at the content. What is the SINGLE best Schema.org Type for this page?
           - If it's a Bio -> Suggest "Physician"
           - If it's a Disease info page -> Suggest "MedicalCondition"
           - If it's a Service page -> Suggest "MedicalProcedure" or "Service"
           - If it's a Blog -> Suggest "BlogPosting"

           COMPARE with "Existing Schema". If the specific type is missing, recommend it.
           If the existing schema is already perfect, return "✅ Optimal".

        OUTPUT JSON: {{
            "rating": "...",
            "writing_quality": "...",
            "suggested_desc": "...",
            "schema_prescription": "..."
        }}
        """

        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        return json.loads(response.text)
    except Exception as e:
        return {
            "rating": "Error",
            "writing_quality": "Error",
            "suggested_desc": "",
            "schema_prescription": str(e),
        }


# --- SCORING ---
def calculate_score(data, ai_result):
    score = 100
    reasons = []

    # Technical
    if not data["JSON Valid"]:
        score -= 30
        reasons.append("Broken Schema Syntax (-30)")
    if data["Title"] == "MISSING":
        score -= 20
        reasons.append("Missing Title (-20)")
    if data["H1"] == "MISSING":
        score -= 10
        reasons.append("Missing H1 (-10)")

    # Content
    if data["Echo Score"] > 85:
        score -= 15
        reasons.append("⚠️ Auto-Generated Desc (-15)")

    t_len = len(data["Title"])
    if t_len < 10 or t_len > 70:
        score -= 5
        reasons.append("Bad Title Len (-5)")

    # AI Quality
    if ai_result.get("rating") == "Low":
        score -= 20
        reasons.append("Low Content Relevance (-20)")

    # If AI suggests a specific Schema change (it didn't say "Optimal")
    prescription = ai_result.get("schema_prescription", "")
    if prescription and "Optimal" not in prescription and "Error" not in prescription:
        score -= 10
        reasons.append("Missing Specific Schema (-10)")

    return max(0, score), ", ".join(reasons)


# --- SCRAPER ---
def scrape_seo_data(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (SEO-Auditor)"}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        title = soup.find("title").get_text().strip() if soup.find("title") else "MISSING"
        h1 = soup.find("h1").get_text().strip() if soup.find("h1") else "MISSING"
        meta = soup.find("meta", attrs={"name": "description"})
        meta_desc = meta["content"].strip() if meta and meta.get("content") else "MISSING"

        schemas = []
        valid_json = True
        for s in soup.find_all("script", type="application/ld+json"):
            if s.string:
                try:
                    json.loads(s.string)
                    schemas.append(s.string)
                except json.JSONDecodeError:
                    valid_json = False

        content_area = soup.find(class_="page-content-area")
        if content_area:
            body_text = content_area.get_text(separator=" ").strip()
        else:
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            body_text = soup.get_text(separator=" ").strip()

        echo_score = 0
        if meta_desc != "MISSING" and body_text:
            matcher = difflib.SequenceMatcher(None, meta_desc, body_text[: len(meta_desc) + 50])
            echo_score = matcher.ratio() * 100

        return {
            "Title": title,
            "H1": h1,
            "Meta Description": meta_desc,
            "Schema Raw": schemas,
            "JSON Valid": valid_json,
            "Body Text": body_text,
            "Echo Score": echo_score,
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
    if not csv_file:
        st.warning("Please upload a CSV first.")
        st.stop()

    stringio = io.StringIO(csv_file.getvalue().decode("utf-8-sig"))
    rows = list(csv.DictReader(stringio))

    if not rows:
        st.error("CSV appears empty or has no header row.")
        st.stop()

    url_key, title_key, headers = detect_csv_columns(rows)

    if not url_key:
        st.error(f"Couldn't find a URL column. Your headers are: {headers}")
        st.stop()

    st.caption(
        f"Detected columns → URL: **{url_key}**"
        + (f" | Title: **{title_key}**" if title_key else "")
    )

    results = []
    bar = st.progress(0)
    status = st.empty()

    total_rows = len(rows)
    processed = 0

    for i, row in enumerate(rows):
        csv_title = row.get(title_key, "") if title_key else row.get("Page Title", "")
        url = row.get(url_key, "")

        if not url or str(url).strip() == "":
            continue

        url = str(url).strip()

        # If the URL is missing a scheme, add https://
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        if use_staging and staging_domain:
            from urllib.parse import urlparse

            path = urlparse(url).path
            url = f"https://{staging_domain}{path}"

        display_name = csv_title.strip() if isinstance(csv_title, str) and csv_title.strip() else url
        status.text(f"[{i+1}/{total_rows}] 🕷️ Scraping: {display_name}...")

        time.sleep(0.2)
        data = scrape_seo_data(url)

        processed += 1

        if "Error" in data:
            results.append(
                {
                    "Page Title": display_name,
                    "URL": url,
                    "Score": 0,
                    "Score Log": "",
                    "Current Title": "",
                    "H1 Tag": "",
                    "Current Desc": "",
                    "✨ AI Suggested Desc": "-",
                    "🔍 Found Schema": "",
                    "💊 Rx Schema": "",
                    "Verify": "",
                    "Status": "ERROR",
                    "Error": data["Error"],
                }
            )
        else:
            if not (isinstance(csv_title, str) and csv_title.strip()):
                display_name = data["Title"]

            schema_list = []
            for s in data["Schema Raw"]:
                try:
                    j = json.loads(s)
                    if "@graph" in j:
                        for item in j["@graph"]:
                            schema_list.append(item.get("@type", "Unknown"))
                    else:
                        schema_list.append(j.get("@type", "Unknown"))
                except Exception:
                    pass

            flat_schema = []
            for item in schema_list:
                if isinstance(item, list):
                    flat_schema.extend(item)
                else:
                    flat_schema.append(item)

            ai_feedback = {}
            if use_ai:
                status.text(f"[{i+1}/{total_rows}] 🤖 Analyzing: {display_name}...")
                ai_feedback = analyze_with_gemini(
                    data["Body Text"],
                    {"Title": data["Title"], "Meta Description": data["Meta Description"]},
                    flat_schema,
                    creds,
                )

            final_score, score_log = calculate_score(data, ai_feedback)
            google_test_url = f"https://search.google.com/test/rich-results?url={urllib.parse.quote(url)}"

            results.append(
                {
                    "Page Title": display_name,
                    "URL": url,
                    "Score": int(final_score),
                    "Score Log": score_log,
                    "Current Title": data["Title"],
                    "H1 Tag": data["H1"],
                    "Current Desc": data["Meta Description"],
                    "✨ AI Suggested Desc": ai_feedback.get("suggested_desc", "-"),
                    "🔍 Found Schema": ", ".join(sorted(set(flat_schema))) if flat_schema else "",
                    "💊 Rx Schema": ai_feedback.get("schema_prescription", "-"),
                    "Verify": google_test_url,
                }
            )

        bar.progress((i + 1) / total_rows)

    if processed == 0:
        status.warning(
            "Audit finished but **zero valid URLs** were found.\n\n"
            "Make sure your CSV has a URL column and the URL cells are not blank."
        )
        st.session_state["seo_results"] = []
    else:
        status.success(f"Audit Complete! Processed {processed} row(s).")
        results.sort(key=lambda x: x.get("Score", 0))
        st.session_state["seo_results"] = results

if st.session_state["seo_results"]:
    df = pd.DataFrame(st.session_state["seo_results"])

    def color_rows(val):
        if val == "High" or val == "✅ Optimal":
            return "color: green; font-weight: bold"
        if val == "Low":
            return "color: red; font-weight: bold"
        return ""

    def color_score(val):
        if isinstance(val, (int, float)):
            if val >= 90:
                return "background-color: #d4edda; color: black; font-weight: bold"
            if val >= 70:
                return "background-color: #fff3cd; color: black; font-weight: bold"
            return "background-color: #f8d7da; color: black; font-weight: bold"
        return ""

    cols = [
        "Page Title",
        "URL",
        "Score",
        "Score Log",
        "Current Title",
        "H1 Tag",
        "Current Desc",
        "✨ AI Suggested Desc",
        "🔍 Found Schema",
        "💊 Rx Schema",
        "Verify",
    ]
    display_cols = [c for c in cols if c in df.columns]

    st.dataframe(
        df[display_cols]
        .style.applymap(color_rows)
        .applymap(color_score, subset=["Score"] if "Score" in df.columns else None),
        column_config={
            "Verify": st.column_config.LinkColumn("Google Tool"),
            "URL": st.column_config.LinkColumn("Live Page"),
            "Score": st.column_config.ProgressColumn("Health", format="%d", min_value=0, max_value=100),
        },
        use_container_width=True,
    )

    csv_bytes = df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Report", csv_bytes, "ai_seo_audit.csv", "text/csv")
