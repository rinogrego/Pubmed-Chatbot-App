# SPRINT

## Initial version

- Single-Page Streamlit
- LangChain stack
  - LangChain Google Gemini for Generative and Embedding Models
  - LangSmith Monitoring
  - LangServe Integration (?)
- Deployment choices
  - HuggingFace Spaces: CPU basic · 2 vCPU · 16 GB · FREE   | Free
  - Railway Hobby Plan: 8 GB RAM / 8 vCPU per Service       | $5/month

## Ideas

- stream mode chat
- display the HumanMessage properly while the LLM is generating text
- Prompt template choices
  - Choose from available options
  - Provide custom system prompt
- deploy on huggingface space
- implement knowledge graph maybe
- display PDF results like PubMed results
- implement feature for chat to be able to cite reference using IEEE style

## Problems

- With proper prompting, LLM can summarize between scraped pubmed paper abstracts, but it doesn't take too long to make the LLM unusable
  - unable to provide answers
  - seems to quick to forget
- Capabilities of chat between more than one uploaded documents still yet to be seen