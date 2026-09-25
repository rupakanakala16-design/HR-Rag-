# 🤖 HR-RAG — Zyro Dynamics HR Assistant

<p align="center">

  <h3 align="center">🧠 AI-Powered HR Assistant using Retrieval-Augmented Generation</h3>

  <p align="center">
    Ask HR questions in natural language and get accurate, document-grounded answers from company policies.
  </p>

</p>

---

## 🌟 About the Project

**HR-RAG** is an AI-powered HR Assistant built using **Retrieval-Augmented Generation (RAG)**.

The system allows employees to ask questions about company policies such as:

- 🏖️ Leave & PTO
- 🏠 Work From Home
- 👥 Employee Handbook
- 💰 Compensation & Benefits
- 📈 Performance Reviews
- 🔐 IT & Data Security
- 📜 Code of Conduct
- 🤝 Onboarding & Separation
- ✈️ Travel & Expenses
- 🛡️ Workplace Harassment Prevention

Instead of relying only on the language model's general knowledge, the application retrieves relevant information from HR policy documents and uses that context to generate grounded answers.

---

## 🎯 Problem Statement

Employees often need information from multiple HR documents.

Finding the correct information manually can be:

- ⏳ Time-consuming
- 📚 Difficult across multiple documents
- 🔍 Hard to search efficiently
- ❌ Prone to misunderstanding

### 💡 Our Solution

HR-RAG provides a conversational AI interface where employees can simply ask:

> 💬 "How do I request PTO?"

The system searches the HR knowledge base, retrieves relevant information, and generates an answer based on the retrieved documents.

---

# ✨ Key Features

### 🤖 AI HR Assistant

Ask HR-related questions using natural language.

### 📚 Document-Based Knowledge

The assistant uses company HR policies and documents as its knowledge base.

### 🔎 Semantic Search

Questions are converted into embeddings and matched against relevant document chunks.

### 🧠 Retrieval-Augmented Generation

Relevant information is retrieved before generating the final response.

### 🛡️ Grounded Answers

The AI is instructed to answer using the retrieved HR context instead of making up information.

### 🚫 Out-of-Scope Protection

If the requested information is not available in the HR documents, the assistant responds with:

> "I can only answer questions based on Zyro Dynamics HR policy documents."

### ⚡ Efficient Processing

The RAG pipeline is cached using Streamlit's resource caching to avoid rebuilding the document and vector pipeline on every interaction.

### 📄 Multiple HR Policy Documents

The system supports a collection of HR and company policy PDFs.

---

# 🧠 How RAG Works

The project follows a complete Retrieval-Augmented Generation pipeline:

```text
             📄 HR POLICY DOCUMENTS
                       │
                       ▼
             📥 PDF DOCUMENT LOADING
                       │
                       ▼
              ✂️ TEXT CHUNKING
             800 chars / 150 overlap
                       │
                       ▼
             🧠 HUGGING FACE
                EMBEDDINGS
                       │
                       ▼
                 🗃️ FAISS
              VECTOR DATABASE
                       │
                       │
                 👤 USER QUESTION
                       │
                       ▼
              🔍 SEMANTIC SEARCH
                  TOP 6 RESULTS
                       │
                       ▼
              📋 RETRIEVED CONTEXT
                       │
                       ▼
             📝 RAG PROMPT
              Context + Question
                       │
                       ▼
              🦙 LLAMA 3.3 70B
                 via Groq
                       │
                       ▼
                💬 AI ANSWER
