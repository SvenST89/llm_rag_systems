# SYSTEM
import pickle
import os
import sys
sys.path.append("/workspaces/20241123_RAG_System/src")
import logging
# DB MANAGEMENT
import chromadb
from chromadb.config import Settings
from uuid import uuid4
# LANGUAGE Modeling
#import faiss
from langchain_community.vectorstores import Chroma
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
# OWN
from chatbackend.openai_api import openai_embedder
from utils.utils import log_function

#===================================================#
# BUILD VECTORSTORE
# Either FAISS or Chroma.
#===================================================#
class VectorStore:
    #===============================#
    # CLASS VARIABLES
    #===============================#
    # Make necessary vectorstore directory
    # Build vectorstore dir
    DATA_DIR = f"/workspaces/20241123_RAG_System/src/data/vectorstore"
    logging.info(f"DATA_DIR: {DATA_DIR}")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    #===============================#
    # INSTANTIATION
    #===============================#
    def __init__(self, vec_db_type: str="chroma", use_case: str="test", embed_mod: str="openai"):
        #self.doclist = doclist
        self.vec_db_type = vec_db_type
        self.use_case = use_case
        self.embed_mod = embed_mod
 
        self.persist_directory = VectorStore.DATA_DIR + '/' + f'persisted_{vec_db_type}_{self.use_case}_index'
 
        if self.vec_db_type == "chroma":
            # INSTANCE VARIABLES
            self.chroma_client_settings = Settings(allow_reset=True, anonymized_telemetry=False)
            # Set up the chormadb client with given settings
            self.chroma_client = chromadb.PersistentClient(path=self.persist_directory, settings=self.chroma_client_settings)
 
        #---------------------------------------#
        # Decision which embedder model is used
        #---------------------------------------#
        if self.embed_mod == "openai":
            self.embedder = openai_embedder()
        else:
            # retrieve first the correct path to the modelhub embedder model
            #embedder_model = 'ModelHub-model-huggingface-intfloat/multilingual-e5-large/main/'
            #dataset_path = "/mnt/imported/data/BNP_PI_DominoRobin_General_Dataset" #get_correct_general_dataset_path()
            embedder_path = "/workspaces/20241123_RAG_System/models/paraphrase-multilingual-MiniLM-L12-v2"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False}
            self.embedder = SentenceTransformerEmbeddings(
                model_name=embedder_path,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
                )
       
    
    #===============================#
    # INSTANCE METHODS.
    #===============================#
    #@log_function
    def build_vectorstore(self, doclist):
        if self.vec_db_type == "chroma":
            db = self.make_chroma_vectorstore(doclist)
        elif self.vec_db_type == "faiss":
            db = self.make_faiss_vectorstore(doclist)
        else:
            raise ValueError("Unknown vector store type chosen! Please use either 'chroma' or 'faiss' as 'vec_db_type'-parameter string!")
            db = None
        return db
 
    # def make_faiss_vectorstore(self, doclist):
    #     '''
    #     Check: https://python.langchain.com/docs/integrations/vectorstores/faiss/
    #     '''
    #     #--------------------------------------------#
    #     # dimensions of SentenceTransformer embedder models - multilingual paraphrase model: 384 https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models
    #     # Dimension of multilingual-35-large model: 1024 https://inference.readthedocs.io/zh-cn/latest/models/builtin/embedding/multilingual-e5-large.html
    #     # Dimension of text-embedding-ada-002 and 3-small: 1536 https://www.pinecone.io/learn/openai-embeddings-v3/
    #     #--------------------------------------------#
    #     uuids = [str(uuid4()) for _ in range(len(doclist))]
    #     # set up InMemory vector store
    #     if self.embed_mod == "openai":
    #         embedding_size = 1536
    #     else:
    #         embedding_size = 384
    #     M = 32 # number of neighbors we add to each vertex on insertion
    #     # to improve search efficiency build a HNSW graph: https://www.pinecone.io/learn/series/faiss/hnsw/
    #     index = faiss.IndexHNSWFlat(embedding_size, M)
    #     inmem_store = FAISS(self.embedder, index, InMemoryDocstore({}), {})
    #     inmem_store.add_documents(doclist, ids=uuids)
    #     inmem_store.save_local(self.persist_directory)
 
    #     return inmem_store
   
    # Reloading FAISS Vectorstore
    def reload_faiss_store(self):
        logging.info("Reloading FAISS Index Vector Store. Check that you match the correct vec_db_type and use_case for reloading!")
        lc_vecstore_db = FAISS.load_local(
            self.persist_directory,
            embeddings=self.embedder,
            allow_dangerous_deserialization=True
        )
        return lc_vecstore_db
 
    def make_chroma_vectorstore(self, doclist):
        """
        This method builds a chroma database with the 'Hierarchical Navigable Small World' (HNSW) algorithm and stores the 'cosine'-similarity metric as function for retrieval.
 
        For more information on how to use ChromaDB, check: https://www.datacamp.com/tutorial/chromadb-tutorial-step-by-step-guide and
        LangChain (https://python.langchain.com/docs/integrations/vectorstores/chroma/#update-and-delete).
        """
        with open(f"{VectorStore.DATA_DIR}/{self.use_case}.pkl", "wb") as file:
            pickle.dump(doclist, file)
        #===============================#
        # VECTORSTORE. CHROMA.
        #===============================#
        # In order to continue with building up a Chroma Database
        # we need to setup our document content, related metadata and ids, individually.
        # Then, making direct use of ChromaDB instead of the LangChain integration
        # enables us to setup a locally stored vector store through flexibly designing our client settings for Chroma.
        # The usual LangChain code `Chroma.from_documents(...)` did **not** store the vector store under the given `persist_directory`.
        # Obviously, the LangChain Chroma-wrapper just caches the data anywhere...
        # Collection Parameters
        collection_params = {
            "hnsw:space": "cosine"
        }
        #db = chroma_client.get_or_create_collection(**collection_params)
        try:
            db = self.chroma_client.create_collection(
                name=self.use_case,
                metadata=collection_params
            )
        except:
            logging.info(f"ChromaDB collection already exists") #- check:\n-----------------------------------------------------------------------\n{e}\n-----------------------------------------------------------------------\nWe will delete and rebuild the DB anew...!")
            self.chroma_client.reset() #.delete_collection(name=self.use_case) does not suffice as there will be duplicates in the db!
            db = self.chroma_client.create_collection(
                name=self.use_case,
                metadata=collection_params
            )
            
        # add documents to chromadb
        uuids = [str(uuid4()) for _ in range(len(doclist))]
        documents, metadatas = self.prepare_data_for_chroma(doclist)
        #print(f"Metadatas for chroma, element 1: {metadatas[1]}")
        embeddings = self.calc_embeddings(documents=documents)

        db.add(
            embeddings = embeddings,
            documents = documents,
            metadatas = metadatas,
            ids = uuids
        )
 
        logging.info(f"Chroma vector database of length {len(db.get(include=['embeddings'])['ids'])} successfully created for use case '{self.use_case}' and stored under:\n{self.persist_directory}")
        return db
 
    # Reloading Chroma Vectorstore
    def reload_chroma_store(self):
        lc_vecstore_db = Chroma(
            client=self.chroma_client,
            collection_name=f"{self.use_case}",
            embedding_function=self.embedder
        )
        return lc_vecstore_db
   
    # add new documents
    def add_docs_to_chroma(self, doclist):
        # https://python.langchain.com/docs/integrations/vectorstores/chroma/#update-and-delete
        vecstore = self.reload_chroma_store()
        uuids = [str(uuid4()) for _ in range(len(doclist))]
        vecstore.add_documents(
            documents = doclist,
            ids = uuids
        )
        return vecstore
   
    #*****************************************
    # Update Documents: Currently not used!!!!
    #*****************************************
    # This would require checking on a section-level of each page, whether the content already exists in the vecstore.
    # If yes, we would need to call the update_documents()-function, otherwise the .add_documents()-function of chroma!
    def update_docs_in_chroma(self, doclist):
        # https://python.langchain.com/docs/integrations/vectorstores/chroma/#update-and-delete
        vecstore = self.reload_chroma_store()
        # get ids that are already in the vectorstore
        ids = vecstore.get()['ids'] # list of ids
 
        vecstore.update_documents(
            documents = doclist,
            ids = ids
        )
 
        return vecstore
   
    # Avoid ChromaDB errors. First transform text into embeddings.
    def calc_embeddings(self, documents):
        embeddings = self.embedder.embed_documents(documents)
        return embeddings
 
    #===============================#
    # INTERNAL METHODS.
    #===============================#
    # prepare texts, metadatas and ids
    @staticmethod
    def prepare_data_for_chroma(doclist):
        documents = [] # store page content inside
        metadatas = [] # store metadata inside
        ids = [] # count ids
        #_id = 0
        for i in range(0, len(doclist)):
            metadat = doclist[i].metadata # metadata dict
            #print(f"Metadata from document {i} from the splitted_document list:\n{metadat}\n--------------------------------------------------------------------------------\n")
            _text = doclist[i].page_content # text string
            #id_ = "id"+f"{_id}"
            #_id+=1
            documents.append(_text)
            metadatas.append(metadat)
            #ids.append(id_)
        return documents, metadatas
 
# usecase="test"
# texts = [
#     "Sven Stoni ist ein super Data Scientist. Er ist 130 Jahre alt und lebt in Piland, Utopia.",
#     "Die beste Data Science Firma ist Polymath Analytics. Diese Firma vertreibt Software-Lösungen zu den Bereichen Economics und Finance.",
#     "Die Produkte von Polymath Analytics befinden sich in einer Preisspanne von 5 - 75€ pro User pro Monat."
#     "Datavation Solutions ist ein dynamisches Start-up, das sich auf maßgeschneiderte Data Science- und Software-Lösungen für Unternehmen jeder Größe spezialisiert hat, von kleinen Start-ups bis hin zu globalen Konzernen.",
#     "Geleitet wird das Unternehmen von Maximilian Brauner, einem visionären CEO, der zuvor bei einem führenden Tech-Riesen in Silicon Valley gearbeitet hat und die Branche mit seinem innovativen Ansatz zur Datenanalyse revolutionieren möchte.",
#     "Im Herzen der Firma arbeitet ein hochkarätiges Team von Data Scientists, darunter Dr. Elina Schmidt, eine Expertin für maschinelles Lernen, die mit ihrem bahnbrechenden Algorithmus zur Vorhersage von Markttrends für Furore sorgte.",
#     "Die Firma bietet ihre maßgeschneiderten Lösungen in verschiedenen Preismodellen an: DataBoost Starter für kleine Unternehmen beginnt bei 499 Euro pro Monat, während die Premium-Lösungen wie PredictPro Enterprise bei 5.000 Euro monatlich liegen.",
#     "Besonders erfolgreich ist der innovative DataStream Dashboard, eine Software, die es Unternehmen ermöglicht, Echtzeitdaten auf beeindruckende Weise zu visualisieren und direkt in ihre Geschäftsprozesse zu integrieren – und das schon ab 1.000 Euro im Monat.",
#     "Unter der Leitung von Sophie Meier, der kreativen CTO des Unternehmens, wurde eine cloudbasierte Plattform entwickelt, die nicht nur Daten speichert, sondern auch durch intelligente Algorithmen eigenständig Erkenntnisse liefert, die Unternehmen bei der Entscheidungsfindung unterstützen.",
#     "Als einer der Hauptinvestoren tritt Hermann Vogel, ein wohlhabender Unternehmer aus der Finanzbranche, auf, der an die transformative Kraft der Datenanalyse glaubt und das Unternehmen mit seiner Expertise in der Skalierung von Tech-Firmen unterstützt.",
#     "Durch den Einsatz neuester KI-Modelle hat Datavation Lösungen entwickelt, die es ihren Kunden ermöglichen, Millionen von Datensätzen in Echtzeit zu verarbeiten, ohne an Rechenkapazität zu verlieren – eine Leistung, die das Unternehmen in der Branche einzigartig macht.",
#     "Der Preis für die Implementierung einer maßgeschneiderten Software-Lösung liegt in der Regel zwischen 20.000 und 100.000 Euro, abhängig von der Komplexität des Projekts und den spezifischen Anforderungen der Kunden.",
#     "Thomas Weber, der verantwortliche Manager für die Kundenbetreuung, sorgt dafür, dass jedes Projekt individuell betreut wird und die Kunden auch nach der Implementierung von Lösungen weiterhin mit maßgeschneiderten Analysen und Support versorgt werden."
#     ]
# docs = []
# for i in range(0, len(texts)):
#     docs.append(Document(page_content=texts[i], metadata={'source': i}))
# vs = VectorStore(vec_db_type="chroma", use_case=usecase, embed_mod="openai")
# db = vs.build_vectorstore(doclist=docs)
# # #dbr = vs.update_docs_in_chroma(doclist=docs)