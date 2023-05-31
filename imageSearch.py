import chromadb
from chromadb.utils import embedding_functions
import pickle
from chromadb.config import Settings
import sys

args = sys.argv

images = []
full_path = "/home/jp/Projects/word-image-search/images"

# {path, description}

class ImageQuery():
    def __init__(self, embeddings=None, refresh=False) -> None:
        documents = {}

        if refresh:
            import imageCaptioner as imageCap
            imageCap.full_path = full_path
            documents = imageCap.get_docs()
        else:
            with open('imageCache.pickle', 'rb') as handle:
                documents = pickle.load(handle)

        client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./chromaDBEmbeddings" # Optional, defaults to .chromadb/ in the current directory
        )
        )

        client.persist()

        if refresh:
            client.reset()


        
        if embeddings == None:
            sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            self.collection = client.get_or_create_collection("documents", embedding_function=sentence_transformer_ef)
        else:
            self.collection = client.create_collection("documents", embedding_function=embeddings)

        # Generate the chromadb collection
        if refresh:
            self.collection.add(
                documents=documents['docs'],
                metadatas=documents['meta'],
                ids=documents['ids'],
            )
        

    def query(self, query):
        results = self.collection.query(
            query_texts=[query],
            n_results=3,
        )

        for data in results['metadatas'][0]:
            print(f"{full_path}/{data['path']}")

a = ImageQuery(refresh=False)
a.query(args[1])