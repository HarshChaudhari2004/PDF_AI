"""Import the packages you'll need:"""

# LangChain components to use
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

# Support for dataset retrieval with Hugging Face
from datasets import load_dataset

# With CassIO, the engine powering the Astra DB integration in LangChain,
# you will also initialize the DB connection:
import cassio

#pip install PyPDF2

from PyPDF2 import PdfReader

"""### Setup"""

ASTRA_DB_APPLICATION_TOKEN = "AstraCS:zMdBgiKWuQfssHZyakrdtRTx:94491c496d78915b971c0091ab0dad6118fcf433a9d0719a8609131b6e48dfb9" # enter the "AstraCS:..." string found in in your Token JSON file
ASTRA_DB_ID = "de50d75f-47cc-4eb1-ac55-939cd157df7c" # enter your Database ID

OPENAI_API_KEY = "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" # enter your OpenAI key

"""#### Provide your secrets:

Replace the following with your Astra DB connection details and your OpenAI API key:
"""

# provide the path of  pdf file/files.
pdfreader = PdfReader('budget_speech.pdf')

from typing_extensions import Concatenate
# read text from pdf
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

raw_text

"""Initialize the connection to your database:

_(do not worry if you see a few warnings, it's just that the drivers are chatty about negotiating protocol versions with the DB.)_
"""

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

"""Create the LangChain embedding and LLM objects for later usage:"""

llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

"""Create your LangChain vector store ... backed by Astra DB!"""

astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="AOA_4Sem",
    session=None,
    keyspace=None,
)

from langchain.text_splitter import CharacterTextSplitter
# We need to split the text using Character Text Split such that it sshould not increse token size
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

texts[:50]

"""### Load the dataset into the vector store


"""

astra_vector_store.add_texts(texts[:50])

print("Inserted %i headlines." % len(texts[:50]))

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)



"""### Run the QA cycle

Simply run the cells and ask a question -- or `quit` to stop. (you can also stop execution with the "▪" button on the top toolbar)

Here are some suggested questions:
- _What is the current GDP?_
- _How much the agriculture target will be increased to and what the focus will be_

"""

first_question = True
while True:
    if first_question:
        query_text = input("\nEnter your question (or type 'quit' to exit): ").strip()
    else:
        query_text = input("\nWhat's your next question (or type 'quit' to exit): ").strip()

    if query_text.lower() == "quit":
        break

    if query_text == "":
        continue

    first_question = False

    print("\nQUESTION: \"%s\"" % query_text)
    answer = astra_vector_index.query(query_text, llm=llm).strip()
    print("ANSWER: \"%s\"\n" % answer)

    print("FIRST DOCUMENTS BY RELEVANCE:")
    for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
        print("    [%0.4f] \"%s ...\"" % (score, doc.page_content[:84]))

