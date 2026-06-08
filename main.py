import os
import json
import chromadb
from typing import List, TypedDict, Annotated, Dict, Any, Union
from pydantic import BaseModel, Field

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

llm = OllamaLLM(model="llama3", temperature=0)
evaluator_llm = OllamaLLM(model="llama3", temperature=0.1)
embeddings = OllamaEmbeddings(model="llama3")

chroma_client = chromadb.HttpClient(
    host='api.trychroma.com', 
    port=443, 
    ssl=True,
    tenant='enter your chromadb tenant_id',  
    database='enter your chromadb database name',                                
    headers={
        "X-Chroma-Token": "enter yoru chromadb api" 
    }
)

degree_db = Chroma(
    client=chroma_client,
    collection_name="usyd_degrees_v2", 
    embedding_function=embeddings
)

memory_db = Chroma(persist_directory="./degree_memory_db", embedding_function=embeddings)

class GraphState(TypedDict):
    question: str
    enhanced_query: str
    memories: str                 
    plan: List[str]               
    past_steps: List[str]         
    documents: List[str]          
    generation: str               
    iteration_count: int
    reward_score: float           
    feedback: str                 
    max_retries: int              

class RewardSignal(BaseModel):
    score: float = Field(description="Score from 0 to 1 based on accuracy and academic policy compliance")
    critique: str = Field(description="Explanation of why the score was given")

def recall_memory_node(state: GraphState):
    print("--- [ACTION] RECALLING EPISODIC MEMORY ---")
    query = state["question"] 
    docs = memory_db.similarity_search(query, k=2)
    memory_context = "\n".join([d.page_content for d in docs])
    return {"memories": memory_context, "iteration_count": 0, "max_retries": 2}

def query_enhancer_node(state: GraphState):
    print("--- [ACTION] ENHANCING USER QUERY FOR RAG ---")
    
    enhancer_prompt = ChatPromptTemplate.from_template(
        "You are an AI Search Optimizer for a university admissions system. "
        "Your job is to convert a casual student query into a formal, highly descriptive search query "
        "optimized for a vector database semantic search.\n\n"
        "Rules:\n"
        "1. Identify the core academic disciplines mentioned (e.g., 'biolog' -> Biological Sciences, Medical Science).\n"
        "2. Include explicit references to entry metrics like ATAR scores if mentioned.\n"
        "3. Output ONLY the final enhanced search string. Do not include introductory text or explanations.\n\n"
        "Raw Query: {question}\n"
        "Past Memory Context: {memories}\n"
        "Enhanced Search Query:"
    )
    
    enhancer_chain = enhancer_prompt | llm | StrOutputParser()
    enhanced_version = enhancer_chain.invoke({
        "question": state["question"],
        "memories": state["memories"]
    })
    
    print(f"-> Original: '{state['question']}'")
    print(f"-> Enhanced: '{enhanced_version.strip()}'\n")
    
    return {"enhanced_query": enhanced_version.strip()}

def planner_node(state: GraphState):
    print("--- [ACTION] STRATEGIZING PLAN ---")
    feedback_context = f"\nPrevious Feedback (Improvement logic): {state.get('feedback', 'None')}"
    
    planner_prompt = ChatPromptTemplate.from_template(
        "You are a University Academic Advisor. Create a 3-step plan to solve the student's query. "
        "The plan MUST be a JSON object with one key 'steps' containing a list of STRINGS only. "
        "{feedback}"
        "Query: {question}\nMemory: {memories}"
    )
    
    planner_chain = planner_prompt | llm | JsonOutputParser()
    plan_output = planner_chain.invoke({
        "question": state["question"], 
        "memories": state["memories"],
        "feedback": feedback_context
    })
    
    steps = plan_output.get("steps", [])
    return {"plan": steps, "past_steps": []}

def execute_step_node(state: GraphState):
    step_idx = len(state["past_steps"])
    current_step = state["plan"][step_idx]
    
    if isinstance(current_step, dict):
        current_step = str(list(current_step.values())[0])
    
    print(f"--- [EXECUTE] Step {step_idx + 1}: {current_step} ---")
    step_text = current_step.lower()
    
    if "search" in step_text or "find" in step_text:
        search_query = state.get("enhanced_query", state["question"])
        docs = degree_db.similarity_search(search_query, k=3)
        doc_texts = [d.page_content for d in docs]
        return {"documents": doc_texts, "past_steps": state["past_steps"] + [current_step]}
    
    return {"past_steps": state["past_steps"] + [current_step]}

def generate_node(state: GraphState):
    print("--- [ACTION] FINAL GENERATION ---")
    gen_prompt = ChatPromptTemplate.from_template(
        "Context: {documents}\nMemory: {memories}\nQuestion: {question}\n"
        "Provide a professional academic advising analysis based on the degree requirements."
    )
    gen_chain = gen_prompt | llm | StrOutputParser()
    
    safe_documents = state.get("documents", [])
    safe_memories = state.get("memories", "No past memories.")
    
    res = gen_chain.invoke({
        "documents": safe_documents,
        "memories": safe_memories,
        "question": state["question"]
    })
    return {"generation": res}

def reward_evaluator_node(state: GraphState):
    print("--- [ADV EVAL] CALCULATING REWARD SIGNAL ---")
    eval_prompt = ChatPromptTemplate.from_template(
        "Evaluate this Academic Advising response for: 1. Accuracy of Degree Info 2. Clarity 3. Professionalism. "
        "Provide a JSON response with 'score' (0.0 to 1.0) and 'critique'.\n"
        "Question: {question}\nResponse: {generation}"
    )
    eval_chain = eval_prompt | evaluator_llm | JsonOutputParser()
    eval_res = eval_chain.invoke({
        "question": state["question"],
        "generation": state["generation"]
    })

    return {
        "reward_score": eval_res.get("score", 0.0),
        "feedback": eval_res.get("critique", ""),
        "iteration_count": state["iteration_count"] + 1
    }

def route_next_action(state: GraphState):
    if len(state["past_steps"]) < len(state["plan"]):
        return "execute_step"
    return "generate"

def route_rl_optimization(state: GraphState):
    reward_data = state.get("reward_score", {})
    
    if isinstance(reward_data, dict):
        actual_score = reward_data.get("score", 0.0)
    else:
        actual_score = getattr(reward_data, "score", 0.0)
        
    iteration = state.get("iteration_count", 0)
    max_retries = state.get("max_retries", 3)

    if actual_score >= 0.8 or iteration >= max_retries:
        print(f"--- [TERMINATE] Reward achieved: {actual_score} ---")
        return "end"
        
    print(f"--- [RE-PLAN] Score too low ({actual_score}). Retrying... ---")
    return "re_plan"

workflow = StateGraph(GraphState)

workflow.add_node("recall_memory", recall_memory_node)
workflow.add_node("query_enhancer", query_enhancer_node) 
workflow.add_node("planner", planner_node)
workflow.add_node("execute_step", execute_step_node)
workflow.add_node("generate", generate_node)
workflow.add_node("evaluate", reward_evaluator_node) 

workflow.set_entry_point("recall_memory")
workflow.add_edge("recall_memory", "query_enhancer")     
workflow.add_edge("query_enhancer", "planner")         
workflow.add_edge("planner", "execute_step")

workflow.add_conditional_edges(
    "execute_step",
    route_next_action,
    {
        "execute_step": "execute_step",
        "generate": "generate" 
    }
)

workflow.add_edge("generate", "evaluate")

workflow.add_conditional_edges(
    "evaluate",
    route_rl_optimization,
    {
        "re_plan": "planner",
        "end": END
    }
)

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "degree_session_001"}}
    user_query = input("What do you want to ask the Academic Advisor? ")
    initial_input = {"question": user_query, "iteration_count": 0}
        
    print("\n--- AGENT STARTING (WITH RL) ---\n")
    for event in app.stream(initial_input, config):
        for node, values in event.items():
            print(f"Node '{node}' completed.")

    state = app.get_state(config)
    
    print("\n--- FINAL ACADEMIC ADVISING ANALYSIS ---\n")
    print(state.values.get("generation", "No generation created."))
