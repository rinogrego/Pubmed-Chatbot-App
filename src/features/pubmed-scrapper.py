from Bio import Entrez
import pandas as pd
import pprint

import argparse

parser = argparse.ArgumentParser(description='Scrap Pubmed Abstracts Given Query')
# parser.add_argument('--query', type=str, help='Specify query for pubmed scrapper', default=None)
parser.add_argument('--retmax', type=int, help="Specify Entrez.esearch's retmax argument", default=None)
parser.add_argument('--target_folder', type=str, help="Specify the folder to save the excel results", default=None)

args = parser.parse_args()
# query = 'high sugar effect' if args.query is None else args.query
retmax = 10 if args.retmax is None else args.retmax
target_folder = "./datasets" if args.target_folder is None else args.target_folder
print_pubmed_article = False

queries = [
    ## self-made
    # "cancer",
    # "diabetes",
    # "cancer SNP",
    # "diabetes SNP",
    # "lifestyle cancer SNP",
    # "lifestyle diabetes SNP",
    # "cancer lifestyle",
    # "diabetes lifestyle",
    # "SNP lifestyle",
    
    # "vitamin SNP",
    # "nutrient SNP",
    # "sugar SNP",
    # "calories SNP",
    # "phenotype SNP nutrient",
    # "genotype SNP nutrient",
    # "PRS lifestyle",
    # "nutrigenomics recommend",
    # "nutrigenomics lifestyle",
    
    # "SNP recommendation",
    # "PRS genomic",
    # "PRS cancer",
    # "PRS vitamin",
    # "PRS nutrient",
    # "PRS sugar",
    # "PRS calories",
    
    ## from ChatGPT
    # "personalized nutrition",
    # "gene-nutrient interactions",
    # "SNP-nutrient interactions",
    # "disease-specific nutrient recommendations",
    # "nutrigenetics",
    # "precision nutrition",
    # "genomic medicine",
    # "genetic variations in nutrient metabolism",
]

import datetime
print("SCRAP PROCESS START:", time_start := datetime.datetime.now())
for query in queries:
    print("QUERY         :", query)
    print("Time          :", q_time_start := datetime.datetime.now())
    print("Retmax        :", retmax)
    print("Target Folder : ", target_folder)

    try:
        # from: https://medium.com/@felipe.odorcyk/scrapping-data-from-pubmed-database-78a9b53de8ca
        def search(query='COVID-19'):
            Entrez.email = 'rinogrego1212@gmail.com'
            handle = Entrez.esearch(
                db='pubmed',
                sort='relevance',
                retstart=0,
                retmax=f'{retmax}',
                retmode='xml',
                term=query
            )
            results = Entrez.read(handle)
            return dict(results)

        def _search(query='COVID-19'):
            Entrez.email = 'rinogrego1212@gmail.com'
            if retmax > 10000:
                handle = {}
                for retmax_chunk in range(0, retmax, 10000):
                    print("Chunk:", retmax_chunk)
                    new_handle = Entrez.esearch(
                        db='pubmed',
                        sort='relevance',
                        retstart=retmax_chunk,
                        retmax=f'{retmax_chunk+10000}',
                        retmode='xml',
                        term=query,
                        use_history="y"
                    )
                    results = Entrez.read(new_handle)
                    handle = {**handle, **dict(results)}
            else:
                handle = Entrez.esearch(
                    db='pubmed',
                    sort='relevance',
                    retstart=0,
                    retmax=f'{retmax}',
                    retmode='xml',
                    term=query
                )
                results = Entrez.read(handle)
            return dict(results)

        studies = search(query)
        translation_set = studies["TranslationSet"]
        query_translation = studies["QueryTranslation"]
        # print("dict(studies):")
        # pprint.pprint(dict(studies))
        # print("\n")
        print("dict(studies).key():")
        pprint.pprint(dict(studies).keys())
        print("studies['Count']:", studies["Count"])
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
        print("\n")
        studiesIdList = studies['IdList']

        def fetch_details(id_list):
            ids = ','.join(id_list)
            Entrez.email = 'rinogrego1212@gmail.com'
            handle = Entrez.efetch(
                db='pubmed',
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
        # print("Keys:\n", dict(studies).keys())
        # print("  Keys of PubmedArticle:\n  ", dict(studies['PubmedArticle']).keys())
        # print("  Length of PubmedArticle:", len(dict(studies)['PubmedArticle']))
        if print_pubmed_article:
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
        # print("  Keys of PubmedBookArticle:\n  ", dict(studies)['PubmedBookArticle'])
        # print("\n\n")
        chunk_size = 100
        for chunk_i in range(0, len(studiesIdList), chunk_size):
            print("Chunk: {} / {}".format(chunk_i, len(studiesIdList)))
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
        if df.shape[0] >= 5:
            print(df.sample(5))
        else:
            print(df)
        print(df.info())

        import os
        query_filename = query.replace(" ", "-")
        filename = f"{query_filename}_pubmed-abstracts.xlsx"
        filepath = os.path.join(target_folder, filename)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        df.to_excel(filepath, index=False)
    except:
        print("Process Failed")
        print("Query            :", query)
    print("Time Taken       :", datetime.datetime.now() - q_time_start)
    print("\n\n" + "="*100 + "\n\n")

print("\nQUERY PROCESS FINISHED")
print("Time Taken       :", datetime.datetime.now() - time_start)
