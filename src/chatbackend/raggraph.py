from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain_community.retrievers import BM25Retriever
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from typing import List, Sequence, Literal
from typing_extensions import TypedDict, Annotated
import logging
import pickle
import os
import re
import sys
sys.path.append("/workspaces/20241123_RAG_System/src")
 
from chatbackend.graph_dataclasses import GradeDocuments, GradeHallucinations, GradeAnswer
from chatbackend.prompts import *
from chatbackend.openai_api import openai_chat_llm
from index.vectorstore import VectorStore
from utils.utils import log_function, logging_decorator

 
#==============================================#
# RETRIEVAL Setup
#==============================================#
# Make a EnsembleRetriever to make hybrid search possible with BM25 keyword-algo (advanced TF-IDF, Best Match 25)
# and chroma-vectorstore semantic retrieval via cosine-similarity and HNSW-algo.
# Check: https://python.langchain.com/docs/how_to/ensemble_retriever/
# it requires `pip install rank_bm25` in your environment!
#-------------------
# Semantic Retriever
use_case = "test"
db = VectorStore("chroma", use_case, embed_mod="openai")
vecstore = db.reload_chroma_store()
semantic_retriever = vecstore.as_retriever(search_kwargs={"k": 7})
#-------------------
# Keyword Retriever: BM25

with open(f'/workspaces/20241123_RAG_System/src/data/vectorstore/{use_case}.pkl', 'rb') as file:
    # Call load method to deserialize
    docs = pickle.load(file)
keyword_retriever = BM25Retriever.from_documents(
    docs
)
keyword_retriever.k = 3
#-------------------
# initialize the ensemble retriever
# check: https://python.langchain.com/api_reference/langchain/retrievers/langchain.retrievers.ensemble.EnsembleRetriever.html
# you can also do weighted reciprocal rank fusion RRF, a simple method for combining the document rankings from multiple IR systems
retriever = EnsembleRetriever(
    retrievers=[keyword_retriever, semantic_retriever], weights=[0.4, 0.6]
)
 
#docs = retriever.invoke("Kartenverlust")
#print(docs)
 
 
#==============================================#
# LLMs Definition
#==============================================#
llm = openai_chat_llm(deployment_name="gpt-4o-mini", maxtokens=3500, temperature=0)
llm_fast = openai_chat_llm(deployment_name="gpt-4o-mini", maxtokens=3500, temperature=0)


#==============================================#
# Document Compression Definition
#==============================================#
compressor = LLMChainFilter.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

#==============================================#
# AGENT Templates
#==============================================#
structured_llm_grader = llm_fast.with_structured_output(GradeDocuments)
retrieval_grader = grade_prompt | structured_llm_grader
 
### Question answering generation
rag_chain = qa_prompt | llm | StrOutputParser()
 
### Hallucination grader
structured_llm_grader = llm.with_structured_output(GradeHallucinations)
hallucination_grader = hallucination_prompt | structured_llm_grader
#==============================================#
# GRAPH Definition
#==============================================#
# Check LangGraph Glossary for explanation of individual parts:
# https://langchain-ai.github.io/langgraph/concepts/low_level/
 
# gloabl app, config
app, config = None, None

### Graph state
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        generation_prompt: The final generation prompt
        messages: List of previous messages
        retry_count: Retry attempts
    """
    question: str
    generation: str
    documents: List[str]
    generation_prompt: str
    #messages: List[BaseMessage]
    doc_history: List[List]
    retry_count: int = 0
    #summary: str
#==============================================#
# GRAPH Nodes
#==============================================#
# Nodes are typically python functions (sync or async) where the first positional argument is the state,
# and (optionally), the second positional argument is a "config", containing optional configurable parameters (such as a thread_id).
 
def retrieve(state: GraphState):
    """
    Retrieve documents
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New keys added to state, documents and doc_history, that contains retrieved documents and the doc history.
    """
    logging.info("---RETRIEVE---")
    question = state["question"]
    # Retrieval
    # If Ensemble Retriever use .invoke instead, if langchain-core >= 0.1.46
    documents = compression_retriever.invoke(question)
    
    try:
        doc_history = state['doc_history']
    except KeyError as e:
        logging.info(f"No doc history yet!\nCheck:\n{e}")
        doc_history = []
    # add docs to doc_history
    doc_history.append(documents)
 
    return {"documents": documents, "question": question, "doc_history": doc_history}
 
def generate(state: GraphState):
    """
    Generate answer.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New keys added to state, generation, generation prompt for logging purposes and all messages.
    """
    logging.info("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    try:
        messages = state["messages"]
    except KeyError as e:
        logging.info(f"No doc history yet!\nCheck:\n{e}")
        messages = []
    doc_history = state['doc_history']
    # RAG generation
    # for logging purposes extract the final prompt
    generation_prompt = rag_chain.get_prompts()[0].format(context=documents, question=question, messages=messages)
    generation = rag_chain.invoke({"context": documents, "question": question, "messages": messages})
    logging.info(generation)
    messages.extend([HumanMessage(content=question), AIMessage(content=generation)])
    return {"documents": documents, "question": question, "generation": generation, "generation_prompt": generation_prompt, "messages": messages, "doc_history": doc_history}

def no_context_answer(state: GraphState):
    """
    Adds a default answer if no context is found.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, generation, that contains the default answer
    """
    DEFAULT_ANSWER = "Wir konnten keinen passenden Kontext für Ihre Frage finden. Könnten Sie bitte weitere Details oder spezifischere Informationen bereitstellen?"
    state['generation'] = DEFAULT_ANSWER
    doc_history = state['doc_history']
    docs = []
    doc_history.append(docs)
    state['documents'] = docs
    state['doc_history'] = doc_history
 
    return state
 
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    logging.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    doc_history = state['doc_history']
    # Score each doc
    filtered_docs = []
    for d in documents:
        logging.info(d.page_content)
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            logging.info("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            logging.info("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    # replace with the filtered docs
    doc_history[-1] = filtered_docs
    return {"documents": filtered_docs, "question": question, "doc_history": doc_history}
 
def delete_messages(state):
    """
    This function deletes older messages from the state to limit the length of the messages list to a certain length.
    The length of the messages list is determined by the configuration variable 'CHAT_HISTORY_LENGTH'.
    The function also deletes older entries from the 'doc_history' list, but only half as many as from the 'messages' list.
 
    Parameters:
    state (dict): A state dictionary that contains the keys 'messages' and 'doc_history'.
                  Both are lists of messages or document histories respectively.
 
    Returns:
    state (dict): The updated state dictionary with the trimmed 'messages' and 'doc_history' lists.
 
    """
 
    # Extract the messages and the document history from the state
    try:
        messages = state["messages"]
    except KeyError as e:
        logging.info(f"No doc history yet!\nCheck:\n{e}")
        messages = []
    doc_history = state['doc_history']
 
    # Determine the maximum length of the messages list from the configuration
    chat_history_len = int(config['CHAT_HISTORY_LENGTH'])
    # Check if the current length of the messages list exceeds the maximum length
    if len(messages) > chat_history_len:
        # If so, we trim the messages list to the last 'chat_history_len' entries
        messages = messages[-chat_history_len:]
        # And we trim the document history to the last 'chat_history_len // 2' entries
        doc_history = doc_history[-(chat_history_len//2):]
 
        # Update the state with the trimmed lists
        state['messages'] = messages
        state['doc_history'] = doc_history
 
    state['retry_count'] = 0 # reset counter for new input(s)
 
    # Return the updated state
    return state
 
#==============================================#
# GRAPH Edges
#==============================================#
# Edges define how the logic is routed and how the graph decides to stop.
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or return a default question.
    Args:
        state (dict): The current graph state
    Returns:
        str: Binary decision for next node to call
    """
    logging.info("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will return a default answer
        logging.info(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, RETURN DEFAULT ANSWER---"
        )
        return "no_context_answer"
    else:
        # We have relevant documents, so generate answer
        logging.info("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.
    Args:
        state (dict): The current graph state
    Returns:
        str: Decision for next node to call
    """
    logging.info("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score
    # Check hallucination
    if grade == "yes":
        logging.info("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        return "useful"
    else:
        logging.info("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
        return "not useful"
 
 
def increment_retry_count(state):
    state['retry_count'] += 1
 
    logging.info(f"---UPDATED RETRY COUNTER TO {state['retry_count']}---")
 
    return state
 
#===========================================================#
# BUILD THE GRAPH
#===========================================================#
# The just follows the flow we outlined in the figure above.
workflow = StateGraph(GraphState)
#----------------#
# NODES
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("increment_retry_counter", increment_retry_count)
workflow.add_node("no_context_answer", no_context_answer)
workflow.add_node("delete_messages", delete_messages)
 
#----------------#
# EDGES
# Ignore answer on chat history again! Always start with retrieval and check if question can be answered!
# Note: At later state we can add advanced methods.
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
# CONDITIONAL EDGES accept the name of a node and a "routing function" to call after that node is executed.
# e.g., from node "grade_documents" go into function "decide_to_generate" in order to check whether the graded
# documents suffice to generate an answer!
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "no_context_answer": "no_context_answer",
        "generate": "generate",
    },
)
 
workflow.add_edge("no_context_answer", "delete_messages")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "useful": "delete_messages",
        "not useful": "increment_retry_counter",
    },
)
 
workflow.add_conditional_edges(
    "increment_retry_counter",
    lambda state: "generate" if state["retry_count"] < int(os.environ["MAXIMUM_RETRY_COUNT"]) + 1 else "no_context_answer",
    {
        "generate": "generate",
        "no_context_answer": "no_context_answer"
    }
)
# 'delete_messages' is the last function to call in order to manage chat history!
# We need to add this final edge from delete_messages to 'END' in order to bring the Graph to an end!
workflow.add_edge("delete_messages", END)
 
# @logging_decorator
# def build_graph(memory, conf):
#     global app, config
#     app = workflow.compile(checkpointer=memory)
#     config = conf
#     return app
 
@logging_decorator
def test_graph(question=""):
    global app, config
    #PROJECT_ROOT_DIR = os.environ['DOMINO_WORKING_DIR']
    #DB = os.path.join("/mnt/data/1001_GenAI_ConsorsWiki_Chatbot/conversation_data", "checkpoints.sqlite")
    #conn = sqlite3.connect(DB, check_same_thread=False)
 
    # Create langgraph rag
    memory = MemorySaver()
    config = {"configurable": {"thread_id": 42}}
    config['CHAT_HISTORY_LENGTH'] = int(os.environ.get('CHAT_HISTORY_LENGTH', 5))
    app = workflow.compile(checkpointer=memory)
    logging.info("Test question is:\n{}".format(question))
    # Use the Runnable
    final_state = app.invoke(
        {"question": question},
        config=config
    )
    logging.info(f"\n*********************************************\nTest answer is:\n{final_state['generation']}\n*********************************************")

test_graph(question="Wie viel kosten die Produkte von Polymath Analytics?")