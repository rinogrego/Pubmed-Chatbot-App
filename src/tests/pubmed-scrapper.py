from Bio import Entrez
import pandas as pd
import pprint

import argparse

parser = argparse.ArgumentParser(description='Scrap Pubmed Abstracts Given Query')
parser.add_argument('--query', type=str, help='Specify query for pubmed scrapper', default=None)
parser.add_argument('--retmax', type=int, help="Specify Entrez.esearch's retmax argument", default=None)
args = parser.parse_args()
query = 'high sugar effect' if args.query is None else args.query
print("Query    :", query)
retmax = 1000 if args.retmax is None else args.retmax
print("Retmax   :", retmax)

# from: https://medium.com/@felipe.odorcyk/scrapping-data-from-pubmed-database-78a9b53de8ca
def search(query='COVID-19'):
    Entrez.email = 'email@example.com'
    handle = Entrez.esearch(db='pubmed',
        sort='relevance',
        retmax=f'{retmax}',
        # retmax='250000',
        retmode='xml',
        term=query
    )
    results = Entrez.read(handle)
    return results

studies = search(query)
pprint.pprint(dict(studies))
pprint.pprint(dict(studies).keys())
print(len(studies["IdList"]))
studiesIdList = studies['IdList']

def fetch_details(id_list):
    ids = ','.join(id_list)
    Entrez.email = 'email@example.com'
    handle = Entrez.efetch(db='pubmed',
        retmode='xml',
        id=ids
    )
    results = Entrez.read(handle)
    return results

title_list= []
abstract_list=[]
journal_list = []
language_list =[]
pubdate_year_list = []
pubdate_month_list = []
studies = fetch_details(studiesIdList)
chunk_size = 100
for chunk_i in range(0, len(studiesIdList), chunk_size):
    print("chunk: {} / {}".format(chunk_i, len(studiesIdList)))
    chunk = studiesIdList[chunk_i:chunk_i + chunk_size]
    papers = fetch_details(chunk)
    for i, paper in enumerate(papers['PubmedArticle']):
        title_list.append(paper['MedlineCitation']['Article']['ArticleTitle'])
        try:
            abstract_list.append(paper['MedlineCitation']['Article']['Abstract']['AbstractText'][0])
        except:
            abstract_list.append('No Abstract')
        journal_list.append(paper['MedlineCitation']['Article']['Journal']['Title'])
        language_list.append(paper['MedlineCitation']['Article']['Language'][0])
        try:
            pubdate_year_list.append(paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Year'])
        except:
            pubdate_year_list.append('No Data')
        try:
            pubdate_month_list.append(paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Month'])
        except:
            pubdate_month_list.append('No Data')
df = pd.DataFrame(list(zip(
    title_list, abstract_list, journal_list, language_list, pubdate_year_list, pubdate_month_list
)),
columns=[
    'Title', 'Abstract', 'Journal', 'Language', 'Year', 'Month'
])

print(df.sample(5))
df.to_excel(f"{query}-pubmed.xlsx", index=False)