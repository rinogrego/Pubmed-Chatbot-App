# from: https://medium.com/@felipe.odorcyk/scrapping-data-from-pubmed-database-78a9b53de8ca
from Bio import Entrez
import pandas as pd

def search(query, retmax=100):
    # Entrez.email = 'email@example.com'
    Entrez.email = 'rinogrego1212@gmail.com'
    handle = Entrez.esearch(db='pubmed',
        sort='relevance',
        retmax=f"{retmax}",
        retmode='xml',
        term=query
    )
    results = Entrez.read(handle)
    return results

def fetch_details(id_list):
    ids = ','.join(id_list)
    Entrez.email = 'rinogrego1212@gmail.com'
    handle = Entrez.efetch(db='pubmed',
        retmode='xml',
        id=ids
    )
    results = Entrez.read(handle)
    return results

def scrap_pubmed(
    query="high sugar effect", 
    save=False,
    retmax=10
):
    title_list= []
    abstract_list=[]
    journal_list = []
    language_list =[]
    pubdate_year_list = []
    pubdate_month_list = []
    studies = search(query=query, retmax=retmax)
    translation_set = studies["TranslationSet"]
    query_translation = studies["QueryTranslation"]
    studiesIdList = studies['IdList']
    studies = fetch_details(studiesIdList)
    
    chunk_size = retmax//10
    
    import streamlit as st
    st.write(f"<h3>Keywords:</h3>{query}", unsafe_allow_html=True)
    st.write(f"<h3>Translation Set:</h3>", unsafe_allow_html=True)
    for translation in translation_set:
        st.write(f"<p>{translation['From']} => {translation['To']}</p>", unsafe_allow_html=True)
    st.write(f"<h3>Query Translation:</h3>{query_translation}<hr>", unsafe_allow_html=True)
    
    progress_text = "Fetching..."
    bar = st.progress(0, text=progress_text)
    for chunk_i in range(0, len(studiesIdList), chunk_size):
        bar.progress(chunk_i/retmax, text="Fetching... | Chunk: {} / {}".format(chunk_i, len(studiesIdList)))
        
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
                pubdate_year_list.append(paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Year'])
            except:
                pubdate_year_list.append('No Data')
            try:
                pubdate_month_list.append(paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Month'])
            except:
                pubdate_month_list.append('No Data')
    bar.empty()
    df = pd.DataFrame(
        list(zip(
            title_list, abstract_list, journal_list, language_list, pubdate_year_list, pubdate_month_list
        )),
        columns=[
        'Title', 'Abstract', 'Journal', 'Language', 'Year', 'Month'
    ])
    if save:
        df.to_excel(f"{query.replace(' ', '-')}--pubmed.xlsx", index=False)
    return df