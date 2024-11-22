from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

load_dotenv()  # Access the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")


def read_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            print(f"Error reading {pdf}:{e}")
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
    llm = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", temperature=0.3)
    chain = load_qa_chain(
        llm,
        chain_type="stuff",
        prompt=PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        ),
    )
    return chain


def user_input(user_question):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )
    print(response)


# pdf_files = ["AWS Certified Solutions Architect Professional Slides v27.pdf","gfs-sosp2003.pdf"]

# combined_text = read_pdf(pdf_files)

# text_chunks = get_text_chunks(combined_text)

# get_vector_store(text_chunks)

user_question = input("\n Please enter your question:")

user_input(user_question)
