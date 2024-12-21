from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
 
# Grader prompt
grader_system_prompt = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", grader_system_prompt),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)
 
# Question answering prompt
qa_system_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context and the chat history to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Context: {context}\nQuestion: {question}\nAnswer:"),
    ]
)
 
# Answer question based on chat history
ha_system_prompt = """You are an assistant for question-answering tasks. Use the following chat history to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Reference the ID of the message in which your answer is grounded."""
ha_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ha_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Question: {question}\nAnswer:"),
    ]
)
 
 
# Hallucination grader prompt
hallucination_system_prompt = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", hallucination_system_prompt),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)
 
# Answer grader prompt
answer_system_prompt = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_system_prompt),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)
 
# Question Re-writer prompt
re_write_system_prompt = """You are a question re-writer that converts an input question to a better version that is optimized 
for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", re_write_system_prompt),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)
 
REQUIRE_SEARCH = PromptTemplate(input_variables=["chat_history", "final_query"], template=\
"""Given the conversation history and a follow up query, determine if the system should call \
an external search tool to better answer the latest user input.
 
Respond "Yes Search" if:
- Specific details or additional knowledge could lead to a better answer.
- There are new or unknown terms, or there is uncertainty what the user is referring to.
- If reading a document cited or mentioned previously may be useful.
 
Respond "Skip Search" if:
- There is sufficient information in chat history to FULLY and ACCURATELY answer the query
and additional information or details would provide little or no value.
- The query is some task that does not require additional information to handle.
 
--------------
Conversation History:
{chat_history}
--------------
 
Even if the topic has been addressed, if more specific details could be useful, \
respond with "Yes Search".
If you are unsure, respond with "Yes Search".
 
Respond with EXACTLY and ONLY "Yes Search" or "Skip Search"
 
Follow Up Input:
{final_query}
"""
)