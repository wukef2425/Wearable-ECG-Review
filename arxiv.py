import requests
import pandas as pd
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

db_url = "http://export.arxiv.org/api/query"
search_query = (
    "(wearable OR portable) AND (ECG OR electrocardiogram) "
    "AND (machine learning OR deep learning OR neural network) "
    "AND (real-time OR low-power OR embedded system OR edge computing) "
    "AND (arrhythmia OR QRS OR PQRST OR R-peak)"
)

start_year = 2020
end_year = 2025

start = 0
max_results = 50
articles = []
new_papers = []

def get_citation_count(title):
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={title}&fields=title,citationCount&limit=1"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data["data"]:
                return data["data"][0].get("citationCount", 0)
    except Exception as e:
        print("Citation lookup error:", e)
    return None

while True:
    query_url = (f"{db_url}?search_query=all:{search_query.replace(' ', '+')}"
                 f"&start={start}&max_results={max_results}")

    response = requests.get(query_url)
    if response.status_code != 200:
        print("Error fetching data from arXiv")
        break

    root = ET.fromstring(response.content)
    entries = root.findall("{http://www.w3.org/2005/Atom}entry")

    if not entries:
        break

    for entry in entries:
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
        link = entry.find("{http://www.w3.org/2005/Atom}id").text.strip()
        abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        published_date = entry.find("{http://www.w3.org/2005/Atom}published").text
        published_year = int(published_date[:4])

        if start_year <= published_year <= end_year:
            citation_count = get_citation_count(title)
            is_recent = datetime.strptime(published_date, "%Y-%m-%dT%H:%M:%SZ") > (datetime.now() - timedelta(days=180))

            if citation_count is not None and citation_count >= 5:
                articles.append([title, link, abstract, published_date, citation_count])

                if is_recent:
                    new_papers.append([title, link, abstract, published_date, citation_count])


            time.sleep(1)

    start += max_results
    time.sleep(2)

df = pd.DataFrame(articles, columns=['Title', 'Link', 'Abstract', 'PublishedDate', 'Citations'])
df.to_csv("arXiv_ECG_ML_with_citations.csv", index=False)

df_new = pd.DataFrame(new_papers, columns=['Title', 'Link', 'Abstract', 'PublishedDate', 'Citations'])
df_new.to_csv("arXiv_ECG_ML_recent_papers.csv", index=False)
