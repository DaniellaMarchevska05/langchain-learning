import os
from platform import system

from dotenv import load_dotenv
from langchain.chains.question_answering.map_reduce_prompt import messages
from langchain.chains.summarize.map_reduce_prompt import prompt_template
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from sqlalchemy import result_tuple
from langchain.schema.runnable import RunnableBranch
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env
load_dotenv()

# model = ChatOpenAI(model = 'gpt-4o-mini')
#

# messages = [
#     SystemMessage(content="Solve the following math problems"),
#     HumanMessage(content="What is 81 divided by 9?"),
# ]

# chat_history = []
#
# system_message = SystemMessage(content='You`re math tutor')
# chat_history.append(system_message)
#
# while True:
#     query = input("You: ")
#     if query.lower() == "exit":
#         break
#     chat_history.append(HumanMessage(content=query))
#
#     result = model.invoke(chat_history)
#     response = result.content
#     chat_history.append(AIMessage(content=response))
#
#     print(f"AI: {response}")
#
# print("---History---")
# print(chat_history)

# PART 2: Prompt with Multiple Placeholders (only HumanMessage in prompt
# template_multiple = """You are a helpful assistant.
# Human: Tell me a {adjective} story about a {animal}.
# Assistant:"""
# prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
# prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"})
# print("\n----- Prompt with Multiple Placeholders -----\n")
# print(prompt)

# #PART 3: Prompt with System and Human Messages (Using Tuples)
# messages = [
#     ("system", "You are a comedian who tells jokes about {topic}."),
#     ("human", "Tell me {joke_count} jokes."),
# ]
# prompt_template = ChatPromptTemplate.from_messages(messages)
# prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
# print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
# print(prompt)


# # PART 1: Create a ChatPromptTemplate using a template string
# template = "Tell be whether there will be explosion if I mix {ingredient} and vinegar"
# prompt_template = ChatPromptTemplate.from_template(template)
# prompt = prompt_template.invoke({"ingredient": "soda"})
# result = model.invoke(prompt)
# print(result.content)

# PART 3: Prompt with System and Human Messages (Using Tuples)
# messages = [
#     ("system", "You`re {subject} tutor. Give brief 1 sentence answers"),
#     ("human", "Tell be whether there will be explosion if I mix {ingredient} and vinegar"),
# ]
#
# prompt_template = ChatPromptTemplate.from_messages(messages)
# prompt = prompt_template.invoke({"subject": "chemistry", "ingredient": "salt"})
# result = model.invoke(prompt)
# print(result.content)

#chains basics instead of prompt_template -> prompt -> result

# prompt_template =ChatPromptTemplate.from_messages(
#     [
#         ("system", "You`re {subject} tutor. Give brief 1 sentence answers"),
#         ("human", "Tell be whether there will be explosion if I mix {ingredient} and vinegar"),
#     ]
# )

#-----------------------runnables--------------------------------------
#for three rennables you don`t need to create RunnableLambda but for more yes
#chain = prompt_template | model | StrOutputParser()

# uppercase_output = RunnableLambda(lambda x: x.upper())
# words_count = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")
#
# chain = prompt_template | model | StrOutputParser() | uppercase_output | words_count
#
# result = chain.invoke({"subject": "chemistry", "ingredient": "salt"})
#
# print(result)


#------------------parallel chains--------------------------------
# prompt_template =ChatPromptTemplate.from_messages(
#     [
#         ("system", "You`re an expert cat food reviewer. Give brief 1 sentence answers. "),
#         ("human", "List the main features of the food {food}"),
#     ]
# )
#
# def analyze_pros(features):
#     pros_template = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You`re an expert cat food reviewer. Give brief 1 sentence answers."),
#             (
#                 "human",
#                 "Given these features: {features}, list the pros of these features.",
#             ),
#         ]
#     )
#     return pros_template.format_prompt(features=features)
#
# def analyze_cons(features):
#     cons_template = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You`re an expert cat food reviewer. Give brief 1 sentence answers."),
#             (
#                 "human",
#                 "Given these features: {features}, list the cons of these features.",
#             ),
#         ]
#     )
#     return cons_template.format_prompt(features=features)
#
#
# # Combine pros and cons into a final review
# def combine_pros_cons(pros, cons):
#     return f"Pros:\n{pros}\n\nCons:\n{cons}"
#
# pros_branch_chain = (
#     RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
# )
#
# cons_branch_chain = (
#     RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
# )
#
#
# chain =(
#     prompt_template
#     | model
#     | StrOutputParser()
#     | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
#     | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
# )
#
# result = chain.invoke({"food": "cottage cheese"})
# print(result)

#--------branching (for adding logic)---------------------------------------------
# Define prompt templates for different feedback types
# positive_feedback_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant."),
#         ("human",
#          "Generate a thank you note for this positive feedback: {feedback}."),
#     ]
# )
#
# negative_feedback_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant."),
#         ("human",
#          "Generate a response addressing this negative feedback: {feedback}."),
#     ]
# )
#
# neutral_feedback_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant."),
#         (
#             "human",
#             "Generate a request for more details for this neutral feedback: {feedback}.",
#         ),
#     ]
# )
#
# escalate_feedback_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant."),
#         (
#             "human",
#             "Generate a message to escalate this feedback to a human agent: {feedback}.",
#         ),
#     ]
# )
#
# # Define the feedback classification template
# classification_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant."),
#         ("human",
#          "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}."),
#     ]
# )
#
# # Define the runnable branches for handling feedback
# branches = RunnableBranch(
#     (
#         lambda x: "positive" in x,
#         positive_feedback_template | model | StrOutputParser()  # Positive feedback chain
#     ),
#     (
#         lambda x: "negative" in x,
#         negative_feedback_template | model | StrOutputParser()  # Negative feedback chain
#     ),
#     (
#         lambda x: "neutral" in x,
#         neutral_feedback_template | model | StrOutputParser()  # Neutral feedback chain
#     ),
#     escalate_feedback_template | model | StrOutputParser()
# )
#
# # Create the classification chain
# classification_chain = classification_template | model | StrOutputParser()
#
# # Combine classification and response generation into one chain
# chain = classification_chain | branches
#
# review = "The product is terrible. It broke after just one use and the quality is very poor."
# result = chain.invoke({"feedback": review})
#
# # Output the result
# print(result)


#------------------RAG basics (retrieval-augmented generation)--------------
#Retrieval-Augmented Generation (RAG) is an AI framework that improves the responses of a language model by retrieving relevant information from an external knowledge source (e.g., a database, documents, or embeddings) before generating a response.
#for one book

# current_dir = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(current_dir, "books", "odyssey.txt")
# persistent_directory = os.path.join(current_dir, "db", "chroma_db")
#
# # Check if the Chroma vector store already exists
# if not os.path.exists(persistent_directory):
#     print("Persistent directory does not exist. Initializing vector store...")
#
#     # Ensure the text file exists
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(
#             f"The file {file_path} does not exist. Please check the path."
#         )
#
#     # Read the text content from the file
#     loader = TextLoader(file_path, encoding="utf-8")
#     documents = loader.load()
#
#     # Split the document into chunks
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#     docs = text_splitter.split_documents(documents)
#
#     # Display information about the split documents
#     print("\n--- Document Chunks Information ---")
#     print(f"Number of document chunks: {len(docs)}")
#     print(f"Sample chunk:\n{docs[0].page_content}\n")
#
#     # Create embeddings
#     print("\n--- Creating embeddings ---")
#     embeddings = OpenAIEmbeddings(
#         model="text-embedding-3-small"
#     )  # Update to a valid embedding model if needed
#     print("\n--- Finished creating embeddings ---")
#
#     # Create the vector store and persist it automatically
#     print("\n--- Creating vector store ---")
#     db = Chroma.from_documents(
#         docs, embeddings, persist_directory=persistent_directory)
#     print("\n--- Finished creating vector store ---")
#
# else:
#     print("Vector store already exists. No need to initialize.")


#-------if embeddings already been created---------------------------------------------
# Define the persistent directory
# current_dir = os.path.dirname(os.path.abspath(__file__))
# persistent_directory = os.path.join(current_dir, "db", "chroma_db")
#
# # Define the embedding model
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#
# # Load the existing vector store with the embedding function
# db = Chroma(persist_directory=persistent_directory,
#             embedding_function=embeddings)
#
#
# # Define the user's question
# query = "Who is Odysseus' wife?"
#
# # Retrieve relevant documents based on the query
# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k": 3, "score_threshold": 0.3},
# )
# relevant_docs = retriever.invoke(query)
#
# # Display the relevant results with metadata
# print("\n--- Relevant Documents ---")
# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}:\n{doc.page_content}\n")
#     if doc.metadata:
#         print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

#---------RAG for many books (retrieval from many books)--------------------------------------------------------------
# Define the directory containing the text files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the books directory exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )

    # List all text files in the directory
    #e.g., ["odyssey.txt", "iliad.txt", ...]
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    # Read the text content from each file and store it with metadata
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path, encoding="utf-8")
        book_docs = loader.load()
        for doc in book_docs:
            # Add metadata (additional info) to each document indicating its source
            doc.metadata = {"source": book_file}  #each chunk in future will contain source
            documents.append(doc)

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

#     it will look like this before splitting: documents = [
#     {"content": "Book 1 full text...", "metadata": {"source": "book1.txt"}},
#     {"content": "Book 2 full text...", "metadata": {"source": "book2.txt"}},
# ]
#
#    it will look like this after splitting: docs = [
#     {"content": "Book 1 - Chunk 1", "metadata": {"source": "book1.txt"}},
#     {"content": "Book 1 - Chunk 2", "metadata": {"source": "book1.txt"}},...

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )  # Update to a valid embedding model if needed
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it
    print("\n--- Creating and persisting vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating and persisting vector store ---")

else:
    print("Vector store already exists. No need to initialize.")

# Define the embedding model in case if embedding were already created
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Define the user's question
query = "Who killed Juliet?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity",  #It finds the most similar documents to the query vector based on cosine similarity.  Use this when you want to retrieve the top k most similar documents.
    search_kwargs={"k": 1},
)
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# Combine the query and the relevant document contents
combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini")

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)