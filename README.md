# Pubmed-Chatbot-App

PubMed paper abstracts chatbot application utilizing Google Gemini's API

## Example

```text
System:
[cari2 prompt medical expert]
[cari2 prompt medical researcher expert]
[cari2 prompt comprehensive explanation]
[cari2 prompt bullet points explanation]

User:
saya mengalami gejala batuk dan pilek. bagaimana cara mengatasinya?

AI:
berikut beberapa metode
```

## Chatbot Features

- medical-specialized prompt
- pubmed scraper for RAG
- pdf-option for context

## RAG Diagram

1. User gives keywords for the backend to search through relevant papers
2. The backend scraps X number of papers, preprocessed into pandas dataframe. The dataframe contains the following columns/criteria:

    - pubmed id
    - title
    - abstract
    - contents
    - background
    - related works
    - methodology
    - experiment result and analysis
    - conclusion
    - other details like journal, authors, etc.

3. The dataframe is filtered based on the keywords using cosine similarity on the embeddings of abstract/contents with regards to the embedding of the keywords.

    - the number of filtered papers can be chosen arbitrarily, but let's say it is X//10

4. Now build the vectorstore. For each row, take its corresponding paper title and abstract/content. And then concatenate it into one single giant string.

    - raw text splitter: `CharacterTextSplitter(separator="/n", chunk_size=2000, chunk_overlap=200, length_function=len)`
    - vectorstore: `FAISS.from_texts(texts=text_chunks, embedding=...)`, where the embedding is either `GoogleGenerativeAIEmbeddings` or `OpenAIEmbeddings`
    - build conversation chain using `ConversationBufferMemory(memory_key="chat_history", return_messages=True)` and `ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)` where LLM is either `ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)` or `ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)`
    - the conversation chain is stored within streamlit's `session_state`

5. The backend then can receive chat message using `st.session_state.conversation({"question": user_query})`

## Problems

- Given a question, retrieving an entire paper can be noisy. The relevant answer might be only in result section or background section.
  - idea:
    - for each chunk after splitting, also provide the title from its source. might be good to also provide the corresponding page
    - design a prompt template that can provide context, title, and page and have the LLM answer based on that context -> reduce token usage
- How to make the LLM give answer while providing the source?
  - idea:
    - design a SYSTEM_PROMPT using few-shot prompt to provide answer and citation

```text
You are good at taking information from a paper's content and providing the source for your answer.
Your answer is always based on bullet points with corresponding source.
You can give comprehensive response in terms of bullet points.
Example:
==============================================================
Question: how to promote healthy lifestyle?
Context:
- paper 1:
    - title: "The Power of Healthy Living: Benefits of Embracing a Balanced Lifestyle"
    - text chunk: Living a healthy lifestyle offers a multitude of benefits that can positively impact every aspect of your life. From physical well-being to mental health and overall quality of life, the advantages are profound. One of the key benefits of a healthy lifestyle is improved physical health. Regular exercise, balanced nutrition, and adequate sleep can help prevent chronic diseases such as heart disease, diabetes, and certain cancers. Maintaining a healthy weight also reduces the risk of developing obesity-related conditions. A healthy lifestyle also has a significant impact on mental health. Physical activity releases endorphins, which are chemicals that improve mood and reduce feelings of stress and anxiety. A balanced diet rich in essential nutrients supports brain health and cognitive function, while proper sleep enhances mood and overall mental well-being. Moreover, embracing a healthy lifestyle can improve your energy levels and boost your immune system. Regular physical activity increases stamina and reduces fatigue, while a balanced diet provides the nutrients needed for optimal immune function. This can lead to fewer illnesses and a faster recovery time when you do get sick. Additionally, a healthy lifestyle can improve your quality of life and longevity. By taking care of your physical and mental health, you can enjoy a higher level of well-being and function better in your daily activities. This can lead to a more fulfilling life and a greater sense of overall happiness.
- paper 2:
    - title: "Balancing Act: The Impact of Work-Life Balance on a Healthy Lifestyle"
    - text chunk: Balancing work and personal life is crucial for maintaining a healthy lifestyle. A well-balanced life allows individuals to fulfill their professional responsibilities while also making time for personal relationships, hobbies, and self-care activities. This balance has several positive impacts on physical, mental, and emotional well-being. Physically, a balanced work life helps reduce stress levels, which can lead to a lower risk of developing chronic illnesses such as heart disease, diabetes, and hypertension. It also allows for more time to engage in physical activities, which are essential for maintaining a healthy weight and overall fitness. Mentally and emotionally, balancing work and personal life can improve mood and overall mental health. It allows individuals to recharge and relax, reducing the risk of burnout and improving overall productivity and job satisfaction. Additionally, spending time with loved ones and engaging in hobbies can provide a sense of fulfillment and purpose, contributing to overall happiness and life satisfaction.
- paper 3:
    - title: "The Benefits of Regular Exercise: A Comprehensive Review"
    - text chunk: Regular exercise plays a crucial role in maintaining a healthy lifestyle, benefiting both physical and mental well-being. Incorporating exercise into your routine can have numerous positive effects on your health. First and foremost, regular exercise helps to maintain a healthy weight. It burns calories and builds muscle, which can help prevent obesity and related health issues such as diabetes, heart disease, and certain cancers. Exercise also contributes to better cardiovascular health by strengthening the heart and improving circulation. This can lower the risk of heart disease, high blood pressure, and stroke. Furthermore, physical activity has been shown to improve mood and mental health. It can reduce symptoms of anxiety and depression, boost self-esteem, and enhance overall cognitive function. Regular exercise also plays a role in improving sleep quality. It can help you fall asleep faster and deepen your sleep, leading to better rest and increased energy levels during the day.
- paper 4:
    - title: "Sustainable Health: The Interconnectedness of Healthy Living and Environmental Well-Being"
    - text chunk: The relationship between a healthy lifestyle and the environment is deeply interconnected. The choices we make regarding our health can have a direct impact on the environment, and vice versa. A healthy lifestyle often involves making sustainable choices that benefit both personal health and the environment. For example, choosing to walk or bike instead of driving reduces carbon emissions and promotes physical activity, benefiting both personal health and the planet. Similarly, eating a diet rich in fruits, vegetables, and whole grains not only supports personal health but also reduces the environmental impact of food production. Plant-based diets generally have a lower carbon footprint compared to diets rich in animal products. Furthermore, reducing waste and recycling can also contribute to a healthier environment and, by extension, a healthier lifestyle. By reducing our consumption and waste, we can help preserve natural resources and reduce pollution, which can have positive effects on both the environment and human health.
- paper 5:
    - title: "Nourish to Flourish: The Importance of Proper Nutrition and Sugar Moderation in a Healthy Lifestyle"
    - text chunk: Proper nutrition is fundamental for maintaining a healthy lifestyle. Eating a balanced diet rich in essential nutrients not only supports overall health but also helps prevent chronic diseases and promotes longevity. One of the key aspects of a healthy diet is to consume a variety of foods to ensure you get all the necessary nutrients. This includes plenty of fruits and vegetables, whole grains, lean proteins, and healthy fats. These foods provide vitamins, minerals, antioxidants, and fiber, which are essential for optimal health. Additionally, it is important to limit the intake of sugary foods and beverages. Excessive sugar consumption has been linked to obesity, type 2 diabetes, heart disease, and other health issues. By reducing sugar intake and opting for healthier alternatives, such as fruits or unsweetened beverages, you can significantly improve your health. Proper hydration is also essential for overall health. Drinking an adequate amount of water helps maintain bodily functions, supports digestion, and regulates body temperature. It is recommended to drink at least eight glasses of water a day, or more depending on your activity level and environment.
Answer: Based on the context provided, here is how we can promote healthy lifestyle
- we should balance our work life [Paper 2]
- exercise more [Paper 3]
- eat proper food and drink [Paper 4, Paper 5]
- don't consume too much sugar [Paper 5]
- taking a walk or bike [Paper 4]
==============================================================
Question: {question}
Context: {context}
Answer:
```
