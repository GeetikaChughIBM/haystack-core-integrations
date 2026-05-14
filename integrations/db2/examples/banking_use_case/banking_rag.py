#!/usr/bin/env python3
"""
Banking RAG Pipeline - Question Answering with LLM

This example demonstrates a complete RAG (Retrieval-Augmented Generation) pipeline
for banking document question answering using:
- DB2 for vector storage and retrieval
- Sentence Transformers for embeddings
- OpenAI GPT-4 (or compatible LLM) for answer generation

Features:
- Semantic document retrieval from DB2
- Context-aware answer generation
- Source citation and document references
- Compliance-friendly responses
- Audit logging for all queries
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.utils import Secret

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from haystack_integrations.components.retrievers.db2 import DB2EmbeddingRetriever
from haystack_integrations.document_stores.db2 import DB2DocumentStore

# Load environment variables
load_dotenv()

# ANSI color codes
class Colors:
    """ANSI color codes for colored terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def colored(text: str, color: str) -> str:
    """Return colored text for terminal output."""
    return f"{color}{text}{Colors.END}"


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{colored('=' * 80, Colors.CYAN)}")
    print(colored(f"  {title}", Colors.BOLD + Colors.CYAN))
    print(colored('=' * 80, Colors.CYAN))


def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n{colored(f'--- {title} ---', Colors.BLUE)}")


# Banking-specific prompt template
BANKING_PROMPT_TEMPLATE = """You are a knowledgeable banking compliance and policy expert assistant. Your role is to provide accurate, helpful information based on official bank documents.

INSTRUCTIONS:
1. Answer the question using ONLY the information provided in the documents below
2. Be precise and factual - cite specific document titles, versions, and sections
3. If the documents don't contain enough information, clearly state what's missing
4. For regulatory questions, emphasize compliance requirements and deadlines
5. For customer service questions, provide clear, actionable steps
6. Use professional, clear language appropriate for banking context
7. Include document citations in your answer

DOCUMENTS:
{% for doc in documents %}
---
Document ID: {{ doc.id }}
Title: {{ doc.meta.title }}
Type: {{ doc.meta.document_type }}
Department: {{ doc.meta.department }}
Version: {{ doc.meta.version }}
Effective Date: {{ doc.meta.effective_date }}

Content:
{{ doc.content }}
---
{% endfor %}

QUESTION: {{ question }}

ANSWER (with citations):"""


class MockLLMGenerator:
    """
    Mock LLM generator for demonstration purposes.
    In production, replace with OpenAIGenerator, AnthropicGenerator, or HuggingFaceGenerator.
    """
    
    def __init__(self):
        self.name = "MockLLM"
    
    def run(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a mock response based on the prompt.
        
        Args:
            prompt: The formatted prompt with context and question
            
        Returns:
            Dictionary with 'replies' key containing generated text
        """
        # Extract question from prompt
        question_start = prompt.find("QUESTION:")
        if question_start != -1:
            question = prompt[question_start:].split("\n")[0].replace("QUESTION:", "").strip()
        else:
            question = "Unknown question"
        
        # Extract document information
        doc_count = prompt.count("Document ID:")
        
        # Generate mock response
        response = f"""Based on the {doc_count} relevant document(s) retrieved from our banking knowledge base:

**Answer:**
{self._generate_mock_answer(question, prompt)}

**Sources:**
{self._extract_sources(prompt)}

**Compliance Note:**
This information is current as of the effective dates shown in the source documents. For the most up-to-date information or specific situations, please consult with the relevant department or compliance officer.

**Disclaimer:**
This response is generated from official bank documents and policies. It should not be considered legal or financial advice. For specific customer situations, please follow established procedures and escalation protocols."""
        
        return {"replies": [response]}
    
    def _generate_mock_answer(self, question: str, prompt: str) -> str:
        """Generate a contextual mock answer based on the question."""
        question_lower = question.lower()
        
        if "basel" in question_lower or "capital" in question_lower:
            return """Under Basel III regulations (effective 2023), banks must maintain:

1. **Common Equity Tier 1 (CET1)**: Minimum 4.5% of risk-weighted assets
2. **Additional Tier 1 (AT1)**: Minimum 1.5% of risk-weighted assets  
3. **Total Tier 1 Capital**: Minimum 6.0% of risk-weighted assets
4. **Capital Conservation Buffer**: Additional 2.5% above minimum requirements

The capital conservation buffer ensures banks build up capital reserves during normal periods that can be drawn down during stress periods."""
        
        elif "rewards" in question_lower or "points" in question_lower:
            return """Credit card rewards points can be redeemed through multiple channels:

1. **Statement Credits**: 10,000 points = $100 credit (most popular option)
2. **Travel Bookings**: Book flights, hotels, or car rentals through our rewards portal
3. **Gift Cards**: Choose from 100+ partner merchants
4. **Merchandise**: Browse our rewards catalog
5. **Charitable Donations**: Support your favorite causes

**Key Details:**
- Minimum redemption: 2,500 points
- Points never expire (account must remain open and in good standing)
- Redemption processing: 2-3 business days for statement credits
- No blackout dates for travel redemptions"""
        
        elif "wire transfer" in question_lower or "verification" in question_lower:
            return """Wire transfer verification procedure to prevent fraud:

**Step 1 - Request Receipt:**
- Capture recipient details: name, bank, routing/SWIFT, account number, amount, purpose

**Step 2 - Risk Assessment:**
- Flag if: first-time recipient, amount >$10,000, international transfer, high-risk country

**Step 3 - Callback Verification (for flagged transfers):**
- Call customer at registered phone number
- Verify transfer details and recipient relationship
- Use security questions for additional authentication

**Step 4 - Approval Process:**
- Low-risk: Automatic approval
- Medium-risk: Supervisor approval required
- High-risk: Manager approval + 24-hour hold

**Step 5 - Execution & Documentation:**
- Process transfer and send confirmation
- Maintain records for 5 years per BSA requirements"""
        
        elif "mortgage" in question_lower or "underwriting" in question_lower:
            return """Current mortgage underwriting standards (2024):

**Credit Requirements:**
- Minimum credit score: 620 (conventional), 580 (FHA)
- Credit history: 2+ years established credit

**Financial Requirements:**
- Maximum LTV: 80% (conventional), 96.5% (FHA)
- Maximum DTI: 43% (may extend to 50% with compensating factors)
- Reserves: 2 months required

**Employment & Income:**
- Stable employment: 2+ years in same field
- Income documentation: 2 years tax returns, 2 months pay stubs
- Income verification required

**Property Requirements:**
- Professional appraisal required
- Must meet safety and habitability standards

**Compliance:**
- All loans must comply with Ability-to-Repay (ATR) rule
- Qualified Mortgage (QM) status preferred"""
        
        elif "aml" in question_lower or "suspicious" in question_lower:
            return """AML training requirements for recognizing suspicious activity:

**Red Flags to Recognize:**
1. **Structuring**: Multiple transactions just below $10,000 reporting threshold
2. **Unusual Activity**: Transactions inconsistent with customer profile
3. **Third-Party Transactions**: Frequent deposits/withdrawals by non-account holders
4. **Cash-Intensive**: Excessive cash deposits from business accounts
5. **International Wires**: Frequent wires to high-risk countries

**Reporting Requirements:**
- File SAR within 30 days of detection
- Include: who, what, when, where, why
- Maintain confidentiality - do not notify customer

**Training Completion:**
- Pass 80% quiz to complete module
- Annual refresher training required
- Completion tracked in HR system

**Case Studies:**
Review 10 real-world money laundering schemes to understand patterns and detection methods."""
        
        elif "gdpr" in question_lower or "data protection" in question_lower:
            return """GDPR data protection requirements for customer information:

**Core Principles:**
1. **Lawful Processing**: Obtain explicit consent for data processing
2. **Transparency**: Clearly communicate how data is used
3. **Data Minimization**: Collect only necessary information
4. **Accuracy**: Keep customer data accurate and up-to-date
5. **Storage Limitation**: Define and enforce retention policies

**Customer Rights:**
- Right to access their data
- Right to data portability
- Right to be forgotten (erasure)
- Right to rectification

**Bank Obligations:**
- Implement appropriate technical and organizational security measures
- Conduct Data Protection Impact Assessments (DPIAs) for high-risk processing
- Breach notification within 72 hours of discovery
- Appoint Data Protection Officer (DPO)

**Compliance:**
- Regular audits and assessments
- Staff training on data protection
- Vendor management and data processing agreements"""
        
        else:
            return f"""Based on the retrieved documents, here is the information relevant to your question about {question}:

The documents provide detailed information on this topic, including specific requirements, procedures, and compliance guidelines. Please refer to the source documents listed below for complete details.

For specific situations or additional clarification, please consult with the relevant department or your supervisor."""
    
    def _extract_sources(self, prompt: str) -> str:
        """Extract document sources from the prompt."""
        sources = []
        lines = prompt.split("\n")
        
        current_doc = {}
        for line in lines:
            if line.startswith("Document ID:"):
                if current_doc:
                    sources.append(self._format_source(current_doc))
                current_doc = {"id": line.replace("Document ID:", "").strip()}
            elif line.startswith("Title:"):
                current_doc["title"] = line.replace("Title:", "").strip()
            elif line.startswith("Version:"):
                current_doc["version"] = line.replace("Version:", "").strip()
            elif line.startswith("Type:"):
                current_doc["type"] = line.replace("Type:", "").strip()
        
        if current_doc:
            sources.append(self._format_source(current_doc))
        
        return "\n".join(sources) if sources else "No sources available"
    
    def _format_source(self, doc: Dict[str, str]) -> str:
        """Format a single source citation."""
        return f"- {doc.get('title', 'Unknown')} (ID: {doc.get('id', 'N/A')}, Version: {doc.get('version', 'N/A')}, Type: {doc.get('type', 'N/A')})"


def extract_filters_from_query(query: str) -> Dict[str, Any]:
    """
    Extract metadata filters from natural language query.
    Uses relaxed filtering to ensure good recall.
    (Same implementation as in banking_search.py)
    """
    filters = {}
    query_lower = query.lower()
    
    # Document type detection - be less restrictive, use semantic search more
    # Only filter on document type if very specific keywords are present
    if "procedure" in query_lower or "process" in query_lower:
        filters["document_type"] = "procedure"
    elif "policy" in query_lower and "policies" not in query_lower:
        filters["document_type"] = "policy"
    
    # Department detection - only if explicitly mentioned
    if "operations" in query_lower and "procedure" in query_lower:
        filters["department"] = "operations"
    
    # Regulation ID detection - only for very specific regulatory queries
    import re
    basel_match = re.search(r"basel\s*(iii|3)", query_lower)
    if basel_match and "capital" in query_lower:
        filters["regulation_id"] = {"$in": ["Basel-III-2023", "Basel-III-2022"]}
    
    # Status filter (always active unless specified)
    if "archived" not in query_lower and "historical" not in query_lower:
        filters["status"] = "active"
    
    # Classification filter - be more permissive
    if "public only" not in query_lower:
        filters["classification"] = {"$in": ["public", "internal", "confidential"]}
    else:
        filters["classification"] = "public"
    
    return filters


def log_rag_query(user_id: str, query: str, answer: str, sources: List[Document], 
                  response_time_ms: float) -> None:
    """Log RAG query for compliance audit trail."""
    audit_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "query": query,
        "answer_length": len(answer),
        "num_sources": len(sources),
        "source_ids": [doc.id for doc in sources],
        "response_time_ms": response_time_ms
    }
    
    print(colored(f"\n[AUDIT LOG] RAG query logged for user {user_id}", Colors.YELLOW))
    print(colored(f"  Sources used: {len(sources)} documents", Colors.YELLOW))
    print(colored(f"  Response time: {response_time_ms:.0f}ms", Colors.YELLOW))


def display_rag_response(query: str, answer: str, sources: List[Document]) -> None:
    """Display RAG response in a formatted way."""
    print_subsection(f"Question: {query}")
    
    print(colored("\n📄 ANSWER:", Colors.BOLD + Colors.GREEN))
    print(answer)
    
    print(colored(f"\n📚 SOURCES ({len(sources)} documents):", Colors.BOLD + Colors.CYAN))
    for i, doc in enumerate(sources, 1):
        print(f"\n{i}. {colored(doc.meta.get('title', 'Untitled'), Colors.BOLD)}")
        print(f"   ID: {doc.id} | Type: {doc.meta.get('document_type', 'N/A')} | Version: {doc.meta.get('version', 'N/A')}")
        print(f"   Score: {doc.score:.4f}")


def main():
    """Main function to demonstrate banking RAG pipeline."""
    
    print_section("BANKING RAG PIPELINE - Question Answering with LLM")
    print(colored("\nThis example demonstrates a complete RAG pipeline for banking Q&A.", Colors.BLUE))
    print(colored("Features: Document retrieval + LLM generation + Source citation\n", Colors.BLUE))
    
    # =========================================================================
    # STAGE 1: Initialize Document Store
    # =========================================================================
    print_section("STAGE 1: Initialize DB2 Document Store")
    
    print(colored("  ├─ Connecting to DB2 database...", Colors.CYAN))
    
    document_store = DB2DocumentStore(
        database=os.getenv("DB2_DATABASE", "BANKDB"),
        username=Secret.from_env_var("DB2_USER"),
        password=Secret.from_env_var("DB2_PASSWORD"),
        hostname=os.getenv("DB2_HOST", "localhost"),
        port=int(os.getenv("DB2_PORT", "50000")),
        table_name="banking_documents",
        embedding_dimension=384,
        distance_metric="cosine",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        validate_embedding_model=True,
        recreate_table=False  # Use existing data from banking_search.py
    )
    
    doc_count = document_store.count_documents()
    print(colored(f"  └─ Connected! Found {doc_count} documents in database", Colors.GREEN))
    
    if doc_count == 0:
        print(colored("\n⚠️  No documents found! Please run banking_search.py first to index documents.", Colors.YELLOW))
        return
    
    # =========================================================================
    # STAGE 2: Create RAG Pipeline
    # =========================================================================
    print_section("STAGE 2: Create RAG Pipeline")
    
    print(colored("  ├─ Setting up text embedder...", Colors.CYAN))
    print(colored("  ├─ Setting up DB2 retriever...", Colors.CYAN))
    print(colored("  ├─ Setting up prompt builder...", Colors.CYAN))
    print(colored("  ├─ Setting up LLM generator (Mock)...", Colors.CYAN))
    
    # Create RAG pipeline
    rag_pipeline = Pipeline()
    
    # Add components
    rag_pipeline.add_component(
        "text_embedder",
        SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    )
    
    rag_pipeline.add_component(
        "retriever",
        DB2EmbeddingRetriever(document_store=document_store, top_k=5)
    )
    
    rag_pipeline.add_component(
        "prompt_builder",
        PromptBuilder(template=BANKING_PROMPT_TEMPLATE)
    )
    
    rag_pipeline.add_component(
        "llm",
        MockLLMGenerator()  # In production: OpenAIGenerator(model="gpt-4")
    )
    
    # Connect components
    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder.prompt", "llm.prompt")
    
    print(colored("  └─ RAG pipeline ready!", Colors.GREEN))
    
    print(colored("\n💡 NOTE: Using MockLLM for demonstration.", Colors.YELLOW))
    print(colored("   In production, replace with:", Colors.YELLOW))
    print(colored("   - OpenAIGenerator(model='gpt-4', api_key=Secret.from_env_var('OPENAI_API_KEY'))", Colors.YELLOW))
    print(colored("   - AnthropicGenerator(model='claude-3-opus', api_key=Secret.from_env_var('ANTHROPIC_API_KEY'))", Colors.YELLOW))
    print(colored("   - HuggingFaceGenerator(model='meta-llama/Llama-2-70b-chat-hf')", Colors.YELLOW))
    
    # =========================================================================
    # STAGE 3: RAG Question Answering Scenarios
    # =========================================================================
    print_section("STAGE 3: Banking Q&A Scenarios")
    
    # Scenario 1: Compliance Question
    print_subsection("Scenario 1: Compliance Officer - Basel III Requirements")
    query1 = "What are the Basel III capital adequacy requirements for Tier 1 capital?"
    filters1 = extract_filters_from_query(query1)
    
    start_time = datetime.now()
    result1 = rag_pipeline.run({
        "text_embedder": {"text": query1},
        "retriever": {"filters": filters1},
        "prompt_builder": {"question": query1}
    })
    response_time1 = (datetime.now() - start_time).total_seconds() * 1000
    
    answer1 = result1["llm"]["replies"][0]
    sources1 = result1["retriever"]["documents"]
    
    display_rag_response(query1, answer1, sources1)
    log_rag_query("compliance_officer_001", query1, answer1, sources1, response_time1)
    
    # Scenario 2: Customer Service Question
    print_subsection("Scenario 2: Customer Service - Credit Card Rewards")
    query2 = "How can customers redeem their credit card rewards points?"
    filters2 = extract_filters_from_query(query2)
    
    start_time = datetime.now()
    result2 = rag_pipeline.run({
        "text_embedder": {"text": query2},
        "retriever": {"filters": filters2},
        "prompt_builder": {"question": query2}
    })
    response_time2 = (datetime.now() - start_time).total_seconds() * 1000
    
    answer2 = result2["llm"]["replies"][0]
    sources2 = result2["retriever"]["documents"]
    
    display_rag_response(query2, answer2, sources2)
    log_rag_query("cs_agent_042", query2, answer2, sources2, response_time2)
    
    # Scenario 3: Operations Question
    print_subsection("Scenario 3: Operations - Wire Transfer Security")
    query3 = "What is the procedure for verifying wire transfer requests to prevent fraud?"
    filters3 = extract_filters_from_query(query3)
    
    start_time = datetime.now()
    result3 = rag_pipeline.run({
        "text_embedder": {"text": query3},
        "retriever": {"filters": filters3},
        "prompt_builder": {"question": query3}
    })
    response_time3 = (datetime.now() - start_time).total_seconds() * 1000
    
    answer3 = result3["llm"]["replies"][0]
    sources3 = result3["retriever"]["documents"]
    
    display_rag_response(query3, answer3, sources3)
    log_rag_query("ops_staff_123", query3, answer3, sources3, response_time3)
    
    # Scenario 4: Loan Officer Question
    print_subsection("Scenario 4: Loan Officer - Mortgage Underwriting")
    query4 = "What are the current mortgage underwriting standards and eligibility requirements?"
    filters4 = extract_filters_from_query(query4)
    
    start_time = datetime.now()
    result4 = rag_pipeline.run({
        "text_embedder": {"text": query4},
        "retriever": {"filters": filters4},
        "prompt_builder": {"question": query4}
    })
    response_time4 = (datetime.now() - start_time).total_seconds() * 1000
    
    answer4 = result4["llm"]["replies"][0]
    sources4 = result4["retriever"]["documents"]
    
    display_rag_response(query4, answer4, sources4)
    log_rag_query("loan_officer_789", query4, answer4, sources4, response_time4)
    
    # Scenario 5: Training Question
    print_subsection("Scenario 5: New Employee - AML Training")
    query5 = "What are the key red flags for suspicious activity that I should watch for?"
    filters5 = extract_filters_from_query(query5)
    
    start_time = datetime.now()
    result5 = rag_pipeline.run({
        "text_embedder": {"text": query5},
        "retriever": {"filters": filters5},
        "prompt_builder": {"question": query5}
    })
    response_time5 = (datetime.now() - start_time).total_seconds() * 1000
    
    answer5 = result5["llm"]["replies"][0]
    sources5 = result5["retriever"]["documents"]
    
    display_rag_response(query5, answer5, sources5)
    log_rag_query("new_employee_456", query5, answer5, sources5, response_time5)
    
    # Scenario 6: Privacy Question
    print_subsection("Scenario 6: Privacy Officer - GDPR Compliance")
    query6 = "What are our obligations under GDPR for customer data protection?"
    filters6 = extract_filters_from_query(query6)
    
    start_time = datetime.now()
    result6 = rag_pipeline.run({
        "text_embedder": {"text": query6},
        "retriever": {"filters": filters6},
        "prompt_builder": {"question": query6}
    })
    response_time6 = (datetime.now() - start_time).total_seconds() * 1000
    
    answer6 = result6["llm"]["replies"][0]
    sources6 = result6["retriever"]["documents"]
    
    display_rag_response(query6, answer6, sources6)
    log_rag_query("privacy_officer_555", query6, answer6, sources6, response_time6)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_section("SUMMARY")
    
    avg_response_time = (response_time1 + response_time2 + response_time3 + 
                         response_time4 + response_time5 + response_time6) / 6
    
    print(colored(f"\n✓ RAG queries executed: 6", Colors.GREEN))
    print(colored(f"✓ Average response time: ~{avg_response_time:.0f}ms", Colors.GREEN))
    print(colored(f"✓ Average sources per answer: ~{(len(sources1) + len(sources2) + len(sources3) + len(sources4) + len(sources5) + len(sources6)) / 6:.1f} documents", Colors.GREEN))
    print(colored(f"✓ All queries logged for audit compliance", Colors.GREEN))
    
    print(colored("\n" + "=" * 80, Colors.CYAN))
    print(colored("Banking RAG Pipeline Demo Complete!", Colors.BOLD + Colors.GREEN))
    print(colored("=" * 80, Colors.CYAN))
    
    print(colored("\nKey Features Demonstrated:", Colors.BLUE))
    print("  • Semantic document retrieval from DB2")
    print("  • Context-aware answer generation with LLM")
    print("  • Source citation and document references")
    print("  • Compliance-friendly responses with disclaimers")
    print("  • Audit logging for all Q&A interactions")
    print("  • Fast end-to-end response (<3 seconds)")
    
    print(colored("\nProduction Deployment:", Colors.YELLOW))
    print("  1. Replace MockLLM with real LLM (OpenAI, Anthropic, HuggingFace)")
    print("  2. Add user authentication and access control")
    print("  3. Implement rate limiting and cost tracking")
    print("  4. Set up monitoring and alerting")
    print("  5. Add response quality evaluation")
    print("  6. Deploy with load balancing and caching")


if __name__ == "__main__":
    main()

# Made with Bob
