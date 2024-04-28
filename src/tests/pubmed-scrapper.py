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
retmax = 10 if args.retmax is None else args.retmax
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
translation_set = studies["TranslationSet"]
query_translation = studies["QueryTranslation"]
print("dict(studies):")
pprint.pprint(dict(studies))
print("\n")
print("dict(studies).key():")
pprint.pprint(dict(studies).keys())
print("\n")
print("Details:")
print(f"len(studies['IdList']): {len(studies['IdList'])}")
print(f"Translation Set: (data type: {type(translation_set)})")
for translation in translation_set:
    pprint.pprint(translation)
# pprint.pprint(translation_set[0])
print()
print(f"Query Translation: (data type: {type(query_translation)})")
pprint.pprint(query_translation)
print("\n\n")
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

title_list = []
abstract_list = []
journal_list = []
language_list = []
pubdate_year_list = []
pubdate_month_list = []
pmid_list = []
doi_list = []
studies = fetch_details(studiesIdList)
print("=========== 'Fetch Details' Results ===========")
# pprint.pprint(dict(studies))
print("Keys:\n", dict(studies).keys())
print("  Keys of PubmedArticle:\n  ", dict(studies['PubmedArticle']).keys())
print("  Length of PubmedArticle:", len(dict(studies)['PubmedArticle']))
for article in dict(studies)["PubmedArticle"]:
    print(article["MedlineCitation"].keys())
    print(f"GeneralNote == {article['MedlineCitation']['GeneralNote']}")
    print(f"OtherAbstract == {article['MedlineCitation']['OtherAbstract']}")
    print(f"CitationSubset == {article['MedlineCitation']['CitationSubset']}")
    print(f"KeywordList == {article['MedlineCitation']['KeywordList']}")
    # print(f"SpaceFlightMission == {article['MedlineCitation']['SpaceFlightMission']}")
    print(f"OtherID == {article['MedlineCitation']['OtherID']}")
    print(f"PMID == {type(article['MedlineCitation']['PMID'])}")
    print(f"DateCompleted == {article['MedlineCitation']['DateCompleted']}")
    print(f"DateRevised == {article['MedlineCitation']['DateRevised']}")
    print(f"Article == {article['MedlineCitation']['Article'].keys()}")
    print(f"Article.ELocationID == {article['MedlineCitation']['Article']['ELocationID']}")
    for elocid in article['MedlineCitation']['Article']['ELocationID']:
        if elocid.attributes['EIdType'] == 'doi':
            print(f"Article.ELocationID.doi == {elocid}")
    print("-"*50)
    # for key, val in article["MedlineCitation"]['Article'].items():
    #     print(f"{key} === {val}")
    #     if key == "Abstract":
    #         print("       length of abstract text:", len(val["AbstractText"]))
    #         if len(val["AbstractText"]) > 1:
    #             abstract_text = "\n".join([str(text) for text in val['AbstractText']])
    #             print("===============")
    #             print(abstract_text)
    #             print("===============")
            # print("Abstract TExtttt")
            # for text in val["AbstractText"]:
            #     print(text)
    # print(article["PubmedData"].keys())
print("  Keys of PubmedBookArticle:\n  ", dict(studies)['PubmedBookArticle'])
print("\n\n")
chunk_size = 100
for chunk_i in range(0, len(studiesIdList), chunk_size):
    print("chunk: {} / {}".format(chunk_i, len(studiesIdList)))
    chunk = studiesIdList[chunk_i:chunk_i + chunk_size]
    papers = fetch_details(chunk)
    for i, paper in enumerate(papers['PubmedArticle']):
        title_list.append(paper['MedlineCitation']['Article']['ArticleTitle'])
        try:
            if len(paper['MedlineCitation']['Article']['Abstract']['AbstractText']) == 1:
                abstract_list.append(paper['MedlineCitation']['Article']['Abstract']['AbstractText'][0])
            else:
                abstract_text = "\n".join([str(text) for text in paper['MedlineCitation']['Article']['Abstract']['AbstractText']])
                abstract_list.append(abstract_text)
        except:
            abstract_list.append('No Abstract')
        journal_list.append(paper['MedlineCitation']['Article']['Journal']['Title'])
        language_list.append(paper['MedlineCitation']['Article']['Language'][0])
        try:
            pubdate_year_list.append(int(paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Year']))
        except:
            pubdate_year_list.append('No Data')
        try:
            pubdate_month_list.append(paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Month'])
        except:
            pubdate_month_list.append('No Data')
        pmid_list.append(int(paper['MedlineCitation']['PMID']))
        doi = ''
        for elocid in paper['MedlineCitation']['Article']['ELocationID']:
            if elocid.attributes['EIdType'] == 'doi':
                doi = elocid
                break
        doi_list.append(doi)
df = pd.DataFrame(list(zip(
    title_list, abstract_list, journal_list, language_list, pubdate_year_list, pubdate_month_list, pmid_list, doi_list
)),
columns=[
    'Title', 'Abstract', 'Journal', 'Language', 'Year', 'Month', 'PMID', "DOI"
])

print("=========== Dataframe info ===========")
print(df.shape)
print(df.sample(5))
print(df.info())
df.to_excel(f"{query}-pubmed.xlsx", index=False)