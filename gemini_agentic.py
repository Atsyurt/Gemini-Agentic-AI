# %% [markdown]
# ## Import necessary libraries**

# %%

import os
import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

stream_answer = ""
event_logs = ""

# %% [markdown]
# ## Part1: Load Gemini models
# -***Note:Please input your  gemini api key here***

# %%

# AIzaSyD2OdJppD2b9ajPdpSHSQ8XhUSl-T1d4eA

# search engine
# AIzaSyBpNE3tap-HTTtnTA7MpdvJ7s2sKmMy4DI

if "GOOGLE_API_KEY" not in os.environ:
    # os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
    os.environ["GOOGLE_API_KEY"] = "???"

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


llm_generator = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
# ai_msg = llm.invoke(messages)
# ai_msg
# print(ai_msg)


# %% [markdown]
# ## Part2: Create a Retriver

# %%
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

# Initialize vectorstore

persist_directory = "db_agentic_ai"  # Directory to save the Chroma DB
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# %% [markdown]
# ## Part3: Basic router

# %%
### Router
import json
from langchain_core.messages import HumanMessage, SystemMessage

# The code snippet you provided defines a Router class that helps in routing user questions to either a vectorstore or a web search based on the content of the question. The router_instructions variable contains a prompt that explains the purpose of the router and guides on when to use the vectorstore or web search.
# Prompt
router_instructions = """You are an expert at routing a user question to a vectorstore or web search.

The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.

Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.

Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

# Test router
# test_web_search = llm.invoke(
#     [SystemMessage(content=router_instructions)]
#     + [
#         HumanMessage(
#             content="Who is favored to win the NFC Championship game in the 2024 season?"
#         )
#     ]
# )
# test_web_search_2 = llm.invoke(
#     [SystemMessage(content=router_instructions)]
#     + [HumanMessage(content="What are the models released today for llama3.2?")]
# )
# test_vector_store_3 = llm.invoke(
#     [SystemMessage(content=router_instructions)]
#     + [HumanMessage(content="What are the types of agent memory?")]
# )
# print(test_web_search.content, test_web_search_2.content, test_vector_store_3.content)


# %% [markdown]
# ### Part4: Retrieval Grader

# %%
# Doc grader instructions
doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

# Grader prompt
doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

# # Test
# question = "What is Chain of thought prompting?"
# docs = retriever.invoke(question)
# doc_txt = docs[0].page_content
# print(docs[0])
# doc_grader_prompt_formatted = doc_grader_prompt.format(
#     document=doc_txt, question=question
# )
# result = llm.invoke(
#     [SystemMessage(content=doc_grader_instructions)]
#     + [HumanMessage(content=doc_grader_prompt_formatted)]
# )
# print(result.content)

# %% [markdown]
# ### Part5: Generate

# %%
# Prompt
rag_prompt = """You are an assistant for question-answering tasks. 

Here is the context to use to answer the question:

{context} 

Think carefully about the above context. 

Now, review the user question:

{question}

Provide an answer to this questions using only the above context. 

Use three sentences maximum and keep the answer concise.

Answer:"""


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# # Test
# docs = retriever.invoke(question)
# docs_txt = format_docs(docs)
# rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
# generation = llm_generator.invoke([HumanMessage(content=rag_prompt_formatted)])
# print(generation.content)

# %% [markdown]
# ### Part6: Hallucination Grader
#

# %%
### Hallucination Grader

# Hallucination grader instructions
hallucination_grader_instructions = """

You are a teacher grading a quiz. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader prompt
hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""

# Test using documents and generation from above
# hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
#     documents=docs_txt, generation=generation.content
# )

# #test
# result = llm.invoke(
#     [SystemMessage(content=hallucination_grader_instructions)]
#     + [HumanMessage(content=hallucination_grader_prompt_formatted)]
# )
# print(result.content)

# # %%


# if '"yes"' in result.content:
#     print("good")

# %% [markdown]
# ###  Part7: Answer Grader
#

# %%
### Answer Grader

# Answer grader instructions
answer_grader_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) The STUDENT ANSWER helps to answer the QUESTION

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader prompt
answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""

# Test
question = "What are the vision models released today as part of Llama 3.2?"
answer = "The Llama 3.2 models released today include two vision models: Llama 3.2 11B Vision Instruct and Llama 3.2 90B Vision Instruct, which are available on Azure AI Model Catalog via managed compute. These models are part of Meta's first foray into multimodal AI and rival closed models like Anthropic's Claude 3 Haiku and OpenAI's GPT-4o mini in visual reasoning. They replace the older text-only Llama 3.1 models."

# Test using question and generation from above
answer_grader_prompt_formatted = answer_grader_prompt.format(
    question=question, generation=answer
)

# #test
# result = llm.invoke(
#     [SystemMessage(content=answer_grader_instructions)]
#     + [HumanMessage(content=answer_grader_prompt_formatted)]
# )
# print(result.content)

# # %% [markdown]
# # ### Part8: Web Search
# # -***Google programmable Search engine google api key and search engine id needed needed please include your google programmable seaarch service  api key here***
# #

# %%
### Search
import os

os.environ["GOOGLE_CSE_ID"] = "?"
os.environ["GOOGLE_API_KEY"] = "?"

from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper

search = GoogleSearchAPIWrapper(k=3)

tool = Tool(
    name="I'm Feeling Lucky",
    description="Search Google and return the first result.",
    func=search.run,
)
# test
# tool.run("python")

# %% [markdown]
# ### Part9: We build the above workflow as a graph using LangGraph.
#
#

# %%
import operator
from typing_extensions import TypedDict
from typing import List, Annotated


class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """

    question: str  # User question
    generation: str  # LLM generation
    web_search: str  # Binary decision to run web search
    max_retries: int  # Max number of retries for answer generation
    answers: int  # Number of answers generated
    loop_step: Annotated[int, operator.add]
    documents: List[str]  # List of retrieved documents


# %% [markdown]
# ### Part10: Create Nodes for the graph

# %%
from langchain.schema import Document
from langgraph.graph import END


### Nodes
# def retrieve(state):
#     """
#     Retrieve documents from vectorstore

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): New key added to state, documents, that contains retrieved documents
#     """
#     print("---RETRIEVE---")
#     question = state["question"]

#     # Write retrieved documents to documents key in state
#     documents = retriever.invoke(question)
#     return {"documents": documents}


def retrieve(state):
    """
    Retrieve questions from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    # print("---RETRIEVE---")
    # question = state["question"]

    # # Write retrieved documents to documents key in state
    # documents = retriever.invoke(question)
    # return {"documents": documents}
    global event_logs
    query = state["question"]

    retrieved_docs = vectordb.similarity_search(
        "" + query, filter={"source": "user_query"}, k=1
    )
    context = {"context": retrieved_docs}
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    # event_logs = event_logs + "\n---RETRIVER---"
    # event_logs = event_logs + "\n---DOCS RETRIVED---"
    return {"documents": retrieved_docs}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    global event_logs
    print("---GENERATE---")
    event_logs = event_logs + "\n---GENERATE---"
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    # RAG generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm_generator.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation, "loop_step": loop_step + 1}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    global event_logs
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    event_logs = event_logs + "\n---CHECK DOCUMENT RELEVANCE TO QUESTION---"

    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=d.page_content, question=question
        )
        result = llm.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )

        grade = ""
        if '"yes"' in result.content:
            grade = "yes"
        else:
            grade = "no"
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            event_logs = event_logs + "\n---GRADE: DOCUMENT RELEVANT---"
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            event_logs = event_logs + "\n---GRADE: DOCUMENT NOT RELEVANT---"
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "web_search": web_search}


def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """
    global event_logs
    print("---WEB SEARCH---")
    event_logs = event_logs + "\n---WEB SEARCH---"
    question = state["question"]
    documents = state.get("documents", [])

    # Web search
    docs = tool.invoke({"query": question})
    # web_results = "\n".join([d["content"] for d in docs])
    web_results = "\n" + docs
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents}


### Edges


def route_question(state):
    """
    Route question firstly to  RAG,other implementations can be added.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    global event_logs
    print("---ROUTE QUESTION---")
    event_logs = event_logs + "\n---ROUTE QUESTION---"

    return "vectorstore"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    global event_logs
    print("---ASSESS GRADED DOCUMENTS---")
    event_logs = event_logs + "\n---ASSESS GRADED DOCUMENTS---"
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        event_logs = (
            event_logs
            + "\n---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )

        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        event_logs = event_logs + "\n---DECISION: GENERATE---"
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    global event_logs
    print("---CHECK HALLUCINATIONS---")
    event_logs = event_logs + "\n---CHECK HALLUCINATIONS---"
    global stream_answer
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3)  # Default to 3 if not provided

    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=format_docs(documents), generation=generation.content
    )
    result = llm.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    grade = ""
    if '"yes"' in result.content:
        grade = "yes"
    # grade = json.loads(result.content)["binary_score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        event_logs = (
            event_logs + "\n---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---"
        )
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        event_logs = event_logs + "\n---GRADE GENERATION vs QUESTION---"
        # Test using question and generation from above
        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=question, generation=generation.content
        )
        result = llm.invoke(
            [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )
        grade = ""
        if '"yes"' in result.content:
            grade = "yes"
        # grade = json.loads(result.content)["binary_score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            event_logs = event_logs + "\n---DECISION: GENERATION ADDRESSES QUESTION---"
            print("## Answer:", generation.content)
            stream_answer = generation.content
            return "useful"
        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            event_logs = (
                event_logs + "\n---DECISION: GENERATION DOES NOT ADDRESS QUESTION---"
            )
            return "not useful"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            event_logs = event_logs + "\n---DECISION: MAX RETRIES REACHED---"
            return "max retries"

    elif state["loop_step"] <= max_retries:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        event_logs = (
            event_logs
            + "\n---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---"
        )
        return "not supported"

    else:
        print("---DECISION: MAX RETRIES REACHED---")
        event_logs = event_logs + "\n---DECISION: MAX RETRIES REACHED---"
        return "max retries"


def save_chat_history(state):
    question = state["question"]
    generation = state["generation"]
    global event_logs
    # Open the file in append mode ('a')
    with open("chat_log.txt", "a") as file:
        # Write a new line to the file
        file.write("\nQ*:" + question + "--" + "A*:" + generation.content)
    # also save the chat to db
    vectordb.add_texts(
        texts=[f"Query: {question}\nAnswer: {generation}"],
        metadatas=[{"source": "user_query"}],  # Optional metadata
    )
    print("---QUESTION AND ANSWER SAVED---")
    event_logs = event_logs + "\n---QUESTION AND ANSWER SAVED---"


# %% [markdown]
# ## Part11: Control Flow for the graph

# %%
from langgraph.graph import StateGraph

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("save_chat", save_chat_history)  # Log chat history

# Build graph
workflow.set_conditional_entry_point(
    # Currently it only route question to the vectorstore when query comes
    route_question,
    {
        # "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": "save_chat",
        "not useful": "websearch",
        "max retries": END,
    },
)
workflow.add_edge("save_chat", END)


# # Compile
graph = workflow.compile()
# display(Image(graph.get_graph().draw_mermaid_png()))

# %% [markdown]
# ## Test

# # %%
# inputs = {"question": "What are the types of agent memory?", "max_retries": 1}
# # inputs = {"question": "Who is messi?", "max_retries": 1}
# for event in graph.stream(inputs, stream_mode="values"):
#     print(event)

# %%
inputs = {"question": "Whaat is the java?", "max_retries": 1}
# inputs = {"question": "Who is messi?", "max_retries": 1}
# for event in graph.stream(inputs, stream_mode="values"):
#     print(event)
print("llm multistep reasoning is working and ready now...")


def run_ai_stream(question, max_retries, return_reasoning_steps):
    """
    The function `run_ai_stream` takes a question and maximum number of retries as input, streams the
    question to a graph, and prints the events received in response.

    :param question: The `question` parameter is the question that will be passed to the AI stream for
    processing. It could be any question or query that you want the AI to provide an answer to
    :param max_retries: The `max_retries` parameter in the `run_ai_stream` function specifies the
    maximum number of retries allowed when attempting to stream data from the graph. If the initial
    attempt to stream data fails, the function will retry up to the specified number of times before
    giving up. This parameter helps control the
    :return: The function `run_ai_stream` takes in a question and a maximum number of retries as input
    parameters. It then creates a dictionary `inputs` with the question and a fixed value of 1 for
    `max_retries`. The function then streams the inputs to a graph and prints the events received from
    the stream. Finally, it returns a variable `stream_answer`, which is not defined in the provided
    """
    global event_logs
    event_logs = ""
    inputs = {"question": question, "max_retries": 1}
    # inputs = {"question": "Who is messi?", "max_retries": 1}
    answers = []
    for event in graph.stream(inputs, stream_mode="values"):
        print(event)
        # print(graph.get_state("question"))
        # print(GraphState["question"])
    print(event_logs)
    if return_reasoning_steps:
        return stream_answer, event_logs

    return stream_answer, ""
