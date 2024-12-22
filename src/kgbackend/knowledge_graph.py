# -*-coding:utf-8 -*-
'''
@File    :   knowledge_graph.py
@Time    :   2024/12/22 21:28:00
@Author  :   Sven STEINBAUER
@Version :   1.0
@Contact :   svensteinbauer89@googlemail.com
@Status  :   DEV
@License :   (C)Copyright 2024, Polymath Analytics
@Desc    :   Core class that builds the graph. It uses the LLM-based Named Entity Recoginition pipeline to identify entities and leverages again an LLM to understand concepts / relationships. 
'''

import networkx as nx
from pydantic import BaseModel, Field, ValidationError
from langchain.prompts import PromptTemplate

from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..'))) # Add the parent directory to the path since we work with notebooks
import logging
import json
import subprocess
from typing import List
import nltk
from nltk.stem import WordNetLemmatizer
import spacy

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.utils.ner import LLMNamedEntityRecognizer

# Define the Concepts class
class Concepts(BaseModel):
    #--------------------------------------------
    # What are we doing here?
    # With Pydantic we define a data structure that defines, in turn,
    # how an output from an LLM should look like and what it actually is.
    # So we could instead use a prompt that tells the llm, in this case, to
    # 'return the output as a List of strings that contains the different concepts'.
    # Usually, in LangChain, we use such BaseModel-Classes together with an LLM instance and
    # the method `.with_structured_output()`.
    concepts_list: List[str] = Field(description="List of concepts")

# Define the KnowledgeGraph class
class KnowledgeGraph:
    def __init__(self, entity_types: bool=True):
        """
        Initializes the KnowledgeGraph with a graph, lemmatizer, and NLP model.
        
        Args:
        - entity_types (bool): If standard entity types shall be used (True) or not (False). Defaults to 'True'.
        
        Attributes:
        - graph: An instance of a networkx Graph.
        - lemmatizer: An instance of WordNetLemmatizer.
        - concept_cache: A dictionary to cache extracted concepts.
        - nlp: An instance of a spaCy NLP model.
        - edges_threshold: A float value that sets the threshold for adding edges based on similarity.
        """
        self.graph = nx.Graph()
        self.lemmatizer = WordNetLemmatizer()
        self.concept_cache = {}
        self.nlp = self._load_spacy_model()
        self.edges_threshold = 0.8
        self.entity_types = entity_types

    def build_graph(self, splits, llm, embedding_model):
        """
        Builds the knowledge graph by adding nodes, creating embeddings, extracting concepts, and adding edges.
        
        Args:
        - splits (list): A list of document splits.
        - llm: An instance of a large language model.
        - embedding_model: An instance of an embedding model.
        
        Returns:
        - None
        """
        self._add_nodes(splits)
        embeddings = self._create_embeddings(splits, embedding_model)
        self._extract_concepts(splits, llm)
        self._add_edges(embeddings)
        self._save_graph_locally()

    def _add_nodes(self, splits):
        """
        Adds nodes to the graph from the document splits.
        
        Args:
        - splits (list): A list of document splits.
        
        Returns:
        - None
        """
        for i, split in enumerate(splits):
            self.graph.add_node(i, content=split.page_content)

    def _create_embeddings(self, splits, embedding_model):
        """
        Creates embeddings for the document splits using the embedding model.
        
        Args:
        - splits (list): A list of document splits.
        - embedding_model: An instance of an embedding model.
        
        Returns:
        - numpy.ndarray: An array of embeddings for the document splits.
        """
        texts = [split.page_content for split in splits]
        return embedding_model.embed_documents(texts)

    def _compute_similarities(self, embeddings):
        """
        Computes the cosine similarity matrix for the embeddings.
        
        Args:
        - embeddings (numpy.ndarray): An array of embeddings.
        
        Returns:
        - numpy.ndarray: A cosine similarity matrix for the embeddings.
        """
        return cosine_similarity(embeddings)

    def _load_spacy_model(self):
        """
        Loads the spaCy NLP model, downloading it if necessary.
        
        Args:
        - None
        
        Returns:
        - spacy.Language: An instance of a spaCy NLP model.
        """
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            logging.info("downloading spaCy model via Linux console: python3 -m spacy download...")
            subprocess.run("python3 -m spacy download en_core_web_sm", shell=True)
            #model = spacy.load("de_core_news_sm")
            # locate the model where it is stored with 'model._path' 
            return spacy.load("en_core_web_sm")

    def _run_ner_with_llm(self, content, model_name: str="gpt-4o-mini"):
        ner = LLMNamedEntityRecognizer(model_name=model_name)
        if self.entity_types:
            _entity_types = ["PERSON", "ORGANIZATION", "MILITARY_UNIT", "GPE", "DATE", "TIME"]
        else:
            # extract all possible entities. The LLM decides!!!
            _entity_types = None
        try:
            entities, json_output = ner.extract_entities(text=content, entity_types=_entity_types)
            # logging.info(f"Entities in JSON-Format: {json_output}")
            # logging.info("Entity MILITARY_UNIT:{}".format(entities.MILITARY_UNIT))
            return entities, json_output
        except ValidationError as e:
            raise ValueError(f"Error extracting entities: {e}")
    
    def _extract_concepts_and_entities(self, content, llm):
        """
        Extracts concepts and named entities from the content using spaCy and a large language model.
        
        Args:
        - content (str): The content from which to extract concepts and entities.
        - llm: An instance of a large language model.
        
        Returns:
        - list: A list of extracted concepts and entities.
        """
        if content in self.concept_cache:
            return self.concept_cache[content]
        
        # Extract named entities using spaCy
        # doc = self.nlp(content)
        # logging.info("AVAILABLE ENTITIES of SpaCy: {}".format(doc.ents))
        # named_entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "DATE"]]
        entity_dict, json_output = self._run_ner_with_llm(content=content)
        # transform json_output-string into dict
        named_entities = []
        d = json.loads(json_output)
        for k in d.keys():
            if k != "OTHER":
                #logging.info("Entity key is:{}".format(k))
                entity_list = d[k]
                for i in entity_list:
                    named_entities.append(i)
            else:
                #logging.info("Entity key is 'OTHER'.")
                entity_dict = d[k]
                if entity_dict:
                    for ek in entity_dict.keys():
                        entity_list = entity_dict[ek]
                        for j in entity_list:
                            named_entities.append(j)
        
        # Extract general concepts using LLM
        concept_extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="Extract key concepts (excluding named entities) from the following text:\n\n{text}\n\nKey concepts:"
        )
        concept_chain = concept_extraction_prompt | llm.with_structured_output(Concepts) # apply the pydantic model for the output defined above
        general_concepts = concept_chain.invoke({"text": content}).concepts_list # through the defined pydantic model we are able to retrieve the list of concepts.
        
        # Combine named entities and general concepts
        all_concepts = list(set(named_entities + general_concepts))
        
        self.concept_cache[content] = all_concepts
        return all_concepts

    def _extract_concepts(self, splits, llm):
        """
        Extracts concepts for all document splits using multi-threading.
        
        Args:
        - splits (list): A list of document splits.
        - llm: An instance of a large language model.
        
        Returns:
        - None
        """
        with ThreadPoolExecutor() as executor:
            future_to_node = {executor.submit(self._extract_concepts_and_entities, split.page_content, llm): i 
                              for i, split in enumerate(splits)}
            
            for future in tqdm(as_completed(future_to_node), total=len(splits), desc="Extracting concepts and entities"):
                node = future_to_node[future]
                concepts = future.result()
                self.graph.nodes[node]['concepts'] = concepts

    def _add_edges(self, embeddings):
        """
        Adds edges to the graph based on the similarity of embeddings and shared concepts.
        
        Args:
        - embeddings (numpy.ndarray): An array of embeddings for the document splits.
        
        Returns:
        - None
        """
        similarity_matrix = self._compute_similarities(embeddings)
        num_nodes = len(self.graph.nodes)
        
        for node1 in tqdm(range(num_nodes), desc="Adding edges"):
            for node2 in range(node1 + 1, num_nodes):
                similarity_score = similarity_matrix[node1][node2]
                if similarity_score > self.edges_threshold:
                    shared_concepts = set(self.graph.nodes[node1]['concepts']) & set(self.graph.nodes[node2]['concepts'])
                    edge_weight = self._calculate_edge_weight(node1, node2, similarity_score, shared_concepts)
                    self.graph.add_edge(node1, node2, weight=edge_weight, 
                                        similarity=similarity_score,
                                        shared_concepts=list(shared_concepts))
    def _save_graph_locally(self):
        import pickle
        DATA_DIR = os.path.join(os.getcwd(), 'src/data/knowledge_graph')
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR, exist_ok=True)
        # save graph object to file
        pickle.dump(self.graph, open(f"{DATA_DIR}/knowledge_graph.pkl", "wb"))

    def _calculate_edge_weight(self, node1, node2, similarity_score, shared_concepts, alpha=0.7, beta=0.3):
        """
        Calculates the weight of an edge based on similarity score and shared concepts.
        
        Args:
        - node1 (int): The first node.
        - node2 (int): The second node.
        - similarity_score (float): The similarity score between the nodes.
        - shared_concepts (set): The set of shared concepts between the nodes.
        - alpha (float, optional): The weight of the similarity score. Default is 0.7.
        - beta (float, optional): The weight of the shared concepts. Default is 0.3.
        
        Returns:
        - float: The calculated weight of the edge.
        """
        max_possible_shared = min(len(self.graph.nodes[node1]['concepts']), len(self.graph.nodes[node2]['concepts']))
        normalized_shared_concepts = len(shared_concepts) / max_possible_shared if max_possible_shared > 0 else 0
        return alpha * similarity_score + beta * normalized_shared_concepts

    def _lemmatize_concept(self, concept):
        """
        Lemmatizes a given concept.
        
        Args:
        - concept (str): The concept to be lemmatized.
        
        Returns:
        - str: The lemmatized concept.
        """
        return ' '.join([self.lemmatizer.lemmatize(word) for word in concept.lower().split()])