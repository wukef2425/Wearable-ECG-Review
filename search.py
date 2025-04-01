import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

db_url = "https://pubmed.ncbi.nlm.nih.gov/"
search_query = (
    "term=(\"wearable\"[Title/Abstract] "
    "AND \"ECG\"[Title/Abstract] "
    "AND (\"machine learning\" OR \"deep learning\" OR \"neural network\")[Title/Abstract] "
    "AND (\"real-time\" OR \"low-power\" OR \"embedded system\" OR \"edge computing\")[Title/Abstract] "
    "AND (\"arrhythmia\" OR \"QRS\" OR \"PQRST\" OR \"R-peak\")[Title/Abstract] "
    "AND (2020:2024[dp]))"
)

search_url = f"{db_url}?{search_query.replace(' ', '+')}"
headers = {'User-Agent': 'Mozilla/5.0'}

start_year = 2020
end_year = 2024

articles = []
page = 1

while True:
    paginated_url = f"{search_url}&page={page}"
    response = requests.get(paginated_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    article_tags = soup.find_all('article', class_='full-docsum')
    if not article_tags:
        break

    for article in article_tags:
        title_tag = article.find('a', class_='docsum-title')
        if title_tag:
            title = title_tag.text.strip()
            link = "https://pubmed.ncbi.nlm.nih.gov" + title_tag['href']

            article_response = requests.get(link, headers=headers)
            article_soup = BeautifulSoup(article_response.content, 'html.parser')

            abstract_tag = article_soup.find('div', class_='abstract-content')
            abstract = abstract_tag.text.strip() if abstract_tag else "No abstract available"

            year = None
            year_tag = article_soup.find('span', class_='cit')
            if year_tag:
                year_match = re.search(r'\b(20\d{2})\b', year_tag.text)
                if year_match:
                    year = int(year_match.group(1))

            if year and start_year <= year <= end_year:
                articles.append([title, link, abstract, year])

            time.sleep(1)

    page += 1

df = pd.DataFrame(articles, columns=['Title', 'Link', 'Abstract', 'Year'])
df.to_csv("ECG_machine_learning_SLR_filtered.csv", index=False)