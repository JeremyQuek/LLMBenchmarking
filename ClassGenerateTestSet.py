import os

from langchain_community.document_loaders import PyPDFLoader

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

class GenerateTestSet:
    def __new__(cls, documents, test_size,
                generator_llm=None,
                critic_llm=None,
                embeddings=None,
                distributions={simple: 0.25, reasoning: 0.25, multi_context: 0.25, conditional: 0.25},
                api_key=None):

        #Instantiate new instance upon new
        instance = super(GenerateTestSet, cls).__new__(cls)

        #Set API KEY
        if api_key:
            os.environ[list(api_key.keys())[0]] = list(api_key.values())[0]

        #Set LLMS
        if generator_llm is None:
            generator_llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

        if critic_llm is None:
            critic_llm = ChatOpenAI(model="gpt-3.5-turbo")

        if embeddings is None:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        #Load documents
        documents_loaded = instance.load_documents(documents)

        #Instantiate generator
        generator = TestsetGenerator.from_langchain(
            generator_llm,
            critic_llm,
            embeddings
        )

        #Generate testset
        test_set = generator.generate_with_langchain_docs(
            documents_loaded,
            test_size=test_size,
            distributions=distributions
        )

        return test_set.to_pandas()

    #Static method
    @staticmethod
    def load_documents(documents):
        documents_loaded = []
        for document in documents:
            loader = PyPDFLoader(document)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata['filename'] = doc.metadata['source']
            documents_loaded.extend(loaded_docs)
        return documents_loaded


