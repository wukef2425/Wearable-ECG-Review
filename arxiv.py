import requests
import pandas as pd
import time
import xml.etree.ElementTree as ET

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
        published = entry.find("{http://www.w3.org/2005/Atom}published").text[:4]  # extract year

        if start_year <= int(published) <= end_year:
            articles.append([title, link, abstract, published])

    start += max_results
    time.sleep(2)

df = pd.DataFrame(articles, columns=['Title', 'Link', 'Abstract', 'Year'])
df.to_csv("arXiv_ECG_machine_learning_SLR_filtered.csv", index=False)