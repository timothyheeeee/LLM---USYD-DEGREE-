# 🎓 USYD Academic Advisor: An Agentic RAG System

Let’s be honest: navigating university degree catalogues, ATAR cut-offs, and prerequisite requirements can feel like an absolute maze for students. 

This project is an **Autonomous AI Academic Advisor**. Built using **LangGraph** and **Llama 3**, it uses an advanced Retrieval-Augmented Generation (RAG) architecture to search a live cloud database of University of Sydney (USYD) degrees and provide accurate, policy-compliant guidance.

Rather than just answering questions blindly, this agent is designed to *think*. It remembers past context, plans its approach, queries a live cloud database, and crucially, **evaluates its own answers** before showing them to the user.

---

## 🧠 System Architecture: How It Thinks

This isn't a standard, single-prompt chatbot. It is built as a state machine using LangGraph, executing the following workflow for every user query:

1. **Episodic Memory (`recall_memory`):** The agent first checks its local vector database for past conversations with the student to maintain context (e.g., remembering if they previously mentioned they were failing maths).
2. **Query Enhancer (`query_enhancer`):** Students often speak casually (e.g., *"I'm good at bio"*). This node uses an LLM to translate casual text into a dense, formal search query optimised for the vector database.
3. **The Planner (`planner_node`):** Before taking action, the agent writes a strict 3-step execution plan in JSON.
4. **The Executor (`execute_step`):** The agent connects to a **hosted ChromaDB cloud instance** to search the live USYD degree database using 4096-dimension Llama 3 embeddings.
5. **The Generator (`generate_node`):** Synthesises the retrieved academic data and memory into clear, professional advice.
6. **The Critic (Reinforcement Learning Loop):** An independent evaluator LLM scores the final answer out of 1.0 for accuracy and compliance. If the score is too low, the system rejects the answer and loops back to the Planner to try a different strategy.

---

## 🛠️ Tech Stack

* **Orchestration:** [LangGraph](https://python.langchain.com/v0.1/docs/langgraph/) (State machine & routing)
* **LLM & Embeddings:** [Ollama](https://ollama.com/) (Running `llama3` locally for data privacy and zero-cost inference)
* **Framework:** [LangChain](https://python.langchain.com/) (Prompting & output parsing)
* **Vector Database:** [ChromaDB](https://www.trychroma.com/) (Hosted cloud instance for degree data, local instance for episodic memory)
* **Data Processing:** Pandas (For initial CSV cleaning and metadata extraction)

---

## 🚀 Getting Started

If you want to run this agent locally, follow these steps.

## Prerequisites

Before running this project, ensure you have the following installed and configured on your local machine:

### 1. System Requirements
* **Python:** Version 3.9 or higher.
* **Operating System:** Windows, macOS, or Linux.

### 2. Local LLM Environment (Ollama)
This project relies on [Ollama](https://ollama.com/) to run the large language models locally. 
1. Download and install Ollama from their official website.
2. Pull the required model by running the following command in your terminal:
   ```bash
   ollama pull llama3
