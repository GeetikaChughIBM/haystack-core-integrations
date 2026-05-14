#!/usr/bin/env python3
"""
Banking Document Search - Intelligent Search for Financial Institutions

This example demonstrates how banks can use DB2 + Haystack for:
- Regulatory compliance document search
- Customer service FAQ retrieval
- Risk management document analysis
- Policy and procedure lookup
- Audit trail and compliance tracking

Features:
- Semantic vector search with DB2
- Metadata filtering (department, document type, date range)
- Intelligent query parsing
- Multi-scenario search examples
- Audit logging for compliance
"""

import csv
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from haystack_integrations.components.retrievers.db2 import DB2EmbeddingRetriever
from haystack_integrations.document_stores.db2 import DB2DocumentStore

# Load environment variables
load_dotenv()

# ANSI color codes for terminal output
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


def load_banking_documents_from_csv(csv_path: str) -> List[Document]:
    """
    Load banking documents from CSV file.
    
    Args:
        csv_path: Path to CSV file containing banking documents
        
    Returns:
        List of Haystack Document objects with metadata
    """
    documents = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Create metadata dictionary
            meta = {
                "title": row["title"],
                "document_type": row["document_type"],
                "department": row["department"],
                "effective_date": row["effective_date"],
                "status": row["status"],
                "classification": row["classification"],
                "version": row["version"],
                "language": row["language"],
                "tags": row["tags"].split(",") if row["tags"] else [],
            }
            
            # Add optional fields if present
            if row.get("regulation_id"):
                meta["regulation_id"] = row["regulation_id"]
            if row.get("expiry_date"):
                meta["expiry_date"] = row["expiry_date"]
            if row.get("category"):
                meta["category"] = row["category"]
            if row.get("subcategory"):
                meta["subcategory"] = row["subcategory"]
            
            # Create document
            doc = Document(
                id=row["id"],
                content=row["content"],
                meta=meta
            )
            documents.append(doc)
    
    return documents


def extract_filters_from_query(query: str) -> Dict[str, Any]:
    """
    Extract metadata filters from natural language query.
    Uses relaxed filtering to ensure good recall.
    
    Args:
        query: Natural language search query
        
    Returns:
        Dictionary of filters for DB2 document store
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
    
    # Date range detection - only if explicitly mentioned
    if "2024" in query_lower:
        filters["effective_date"] = {"$gte": "2024-01-01"}
    elif "2023" in query_lower:
        filters["effective_date"] = {"$gte": "2023-01-01", "$lt": "2024-01-01"}
    
    # Regulation ID detection - only for very specific regulatory queries
    basel_match = re.search(r"basel\s*(iii|3)", query_lower)
    if basel_match and "capital" in query_lower:
        # Only filter by regulation_id if asking specifically about Basel III
        filters["regulation_id"] = {"$in": ["Basel-III-2023", "Basel-III-2022"]}
    
    # Status filter (always active unless specified)
    if "archived" not in query_lower and "historical" not in query_lower:
        filters["status"] = "active"
    
    # Classification filter - be more permissive, only exclude restricted
    # Allow internal and confidential documents for better recall
    if "public only" not in query_lower:
        filters["classification"] = {"$in": ["public", "internal", "confidential"]}
    else:
        filters["classification"] = "public"
    
    return filters


def log_query_audit(user_id: str, query: str, filters: Dict[str, Any], 
                    retrieved_docs: List[Document], response_time_ms: float) -> None:
    """
    Log query for compliance audit trail.
    
    Args:
        user_id: User identifier
        query: Search query
        filters: Applied filters
        retrieved_docs: Retrieved documents
        response_time_ms: Query response time in milliseconds
    """
    audit_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "query": query,
        "filters": filters,
        "num_results": len(retrieved_docs),
        "document_ids": [doc.id for doc in retrieved_docs],
        "response_time_ms": response_time_ms
    }
    
    # In production, this would write to audit table in DB2
    # For demo, we'll just print it
    print(colored(f"\n[AUDIT LOG] Query logged for user {user_id}", Colors.YELLOW))


def display_search_results(query: str, documents: List[Document], filters: Dict[str, Any]) -> None:
    """
    Display search results in a formatted way.
    
    Args:
        query: Search query
        documents: Retrieved documents
        filters: Applied filters
    """
    print_subsection(f"Query: {query}")
    
    if filters:
        print(colored(f"Applied Filters: {json.dumps(filters, indent=2)}", Colors.CYAN))
    
    print(colored(f"\nFound {len(documents)} relevant documents:", Colors.GREEN))
    
    for i, doc in enumerate(documents, 1):
        title = doc.meta.get("title", "Untitled")
        print(f"\n{colored(f'{i}. {title}', Colors.BOLD)}")
        print(f"   {colored('ID:', Colors.CYAN)} {doc.id}")
        print(f"   {colored('Type:', Colors.CYAN)} {doc.meta.get('document_type', 'N/A')}")
        print(f"   {colored('Department:', Colors.CYAN)} {doc.meta.get('department', 'N/A')}")
        print(f"   {colored('Version:', Colors.CYAN)} {doc.meta.get('version', 'N/A')}")
        print(f"   {colored('Effective Date:', Colors.CYAN)} {doc.meta.get('effective_date', 'N/A')}")
        print(f"   {colored('Score:', Colors.CYAN)} {doc.score:.4f}")
        
        # Show content preview (first 200 characters)
        content = doc.content or ""
        content_preview = content[:200] + "..." if len(content) > 200 else content
        print(f"   {colored('Preview:', Colors.CYAN)} {content_preview}")


def main():
    """Main function to demonstrate banking document search."""
    
    print_section("BANKING DOCUMENT SEARCH - DB2 + Haystack")
    print(colored("\nThis example demonstrates intelligent document search for financial institutions.", Colors.BLUE))
    print(colored("Use cases: Compliance, Customer Service, Risk Management, Policy Lookup\n", Colors.BLUE))
    
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
        recreate_table=True  # Set to False in production
    )
    
    print(colored("  └─ Document store initialized successfully", Colors.GREEN))
    
    # =========================================================================
    # STAGE 2: Load and Ingest Documents
    # =========================================================================
    print_section("STAGE 2: Load Banking Documents from CSV")
    
    csv_path = Path(__file__).parent / "banking_documents.csv"
    
    if not csv_path.exists():
        print(colored(f"  ✗ CSV file not found: {csv_path}", Colors.RED))
        print(colored("  Please ensure banking_documents.csv exists in the same directory", Colors.YELLOW))
        return
    
    print(colored(f"  ├─ Loading documents from: {csv_path}", Colors.CYAN))
    documents = load_banking_documents_from_csv(str(csv_path))
    print(colored(f"  └─ Loaded {len(documents)} banking documents", Colors.GREEN))
    
    # =========================================================================
    # STAGE 3: Generate Embeddings and Index
    # =========================================================================
    print_section("STAGE 3: Generate Embeddings and Index Documents")
    
    print(colored("  ├─ Initializing embedding model (sentence-transformers/all-MiniLM-L6-v2)...", Colors.CYAN))
    print(colored("  ├─ Generating 384-dimensional vectors...", Colors.CYAN))
    
    # Create indexing pipeline
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(
        "embedder",
        SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    )
    indexing_pipeline.add_component(
        "writer",
        DocumentWriter(document_store=document_store)
    )
    indexing_pipeline.connect("embedder.documents", "writer.documents")
    
    # Run indexing
    indexing_pipeline.run({"embedder": {"documents": documents}})
    
    print(colored(f"  └─ Successfully indexed {len(documents)} documents with embeddings", Colors.GREEN))
    
    # =========================================================================
    # STAGE 4: Create Search Pipeline
    # =========================================================================
    print_section("STAGE 4: Create Search Pipeline")
    
    print(colored("  ├─ Setting up text embedder...", Colors.CYAN))
    print(colored("  ├─ Setting up DB2 embedding retriever...", Colors.CYAN))
    
    search_pipeline = Pipeline()
    search_pipeline.add_component(
        "text_embedder",
        SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    )
    search_pipeline.add_component(
        "retriever",
        DB2EmbeddingRetriever(document_store=document_store, top_k=5)
    )
    search_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    
    print(colored("  └─ Search pipeline ready", Colors.GREEN))
    
    # =========================================================================
    # STAGE 5: Search Scenarios
    # =========================================================================
    print_section("STAGE 5: Banking Search Scenarios")
    
    # Scenario 1: Compliance Officer - Regulatory Search
    print_subsection("Scenario 1: Compliance Officer - Basel III Capital Requirements")
    query1 = "What are the capital adequacy requirements under Basel III for Tier 1 capital?"
    filters1 = extract_filters_from_query(query1)
    
    start_time = datetime.now()
    result1 = search_pipeline.run({
        "text_embedder": {"text": query1},
        "retriever": {"filters": filters1}
    })
    response_time1 = (datetime.now() - start_time).total_seconds() * 1000
    
    display_search_results(query1, result1["retriever"]["documents"], filters1)
    log_query_audit("compliance_officer_001", query1, filters1, 
                    result1["retriever"]["documents"], response_time1)
    
    # Scenario 2: Customer Service - FAQ Lookup
    print_subsection("Scenario 2: Customer Service Agent - Credit Card Rewards")
    query2 = "Customer asking about credit card rewards points redemption options"
    filters2 = extract_filters_from_query(query2)
    
    start_time = datetime.now()
    result2 = search_pipeline.run({
        "text_embedder": {"text": query2},
        "retriever": {"filters": filters2}
    })
    response_time2 = (datetime.now() - start_time).total_seconds() * 1000
    
    display_search_results(query2, result2["retriever"]["documents"], filters2)
    log_query_audit("cs_agent_042", query2, filters2, 
                    result2["retriever"]["documents"], response_time2)
    
    # Scenario 3: Risk Manager - Risk Assessment
    print_subsection("Scenario 3: Risk Manager - Commercial Real Estate Risk")
    query3 = "Show me risk assessments related to commercial real estate credit risk"
    filters3 = extract_filters_from_query(query3)
    
    start_time = datetime.now()
    result3 = search_pipeline.run({
        "text_embedder": {"text": query3},
        "retriever": {"filters": filters3}
    })
    response_time3 = (datetime.now() - start_time).total_seconds() * 1000
    
    display_search_results(query3, result3["retriever"]["documents"], filters3)
    log_query_audit("risk_manager_007", query3, filters3, 
                    result3["retriever"]["documents"], response_time3)
    
    # Scenario 4: Operations - Procedure Lookup
    print_subsection("Scenario 4: Operations Staff - Wire Transfer Verification")
    query4 = "What is the procedure for verifying wire transfer requests to prevent fraud?"
    filters4 = extract_filters_from_query(query4)
    
    start_time = datetime.now()
    result4 = search_pipeline.run({
        "text_embedder": {"text": query4},
        "retriever": {"filters": filters4}
    })
    response_time4 = (datetime.now() - start_time).total_seconds() * 1000
    
    display_search_results(query4, result4["retriever"]["documents"], filters4)
    log_query_audit("ops_staff_123", query4, filters4, 
                    result4["retriever"]["documents"], response_time4)
    
    # Scenario 5: Compliance - AML Training
    print_subsection("Scenario 5: New Employee - AML Training Requirements")
    query5 = "What are the AML training requirements for recognizing suspicious activity?"
    filters5 = extract_filters_from_query(query5)
    
    start_time = datetime.now()
    result5 = search_pipeline.run({
        "text_embedder": {"text": query5},
        "retriever": {"filters": filters5}
    })
    response_time5 = (datetime.now() - start_time).total_seconds() * 1000
    
    display_search_results(query5, result5["retriever"]["documents"], filters5)
    log_query_audit("new_employee_456", query5, filters5, 
                    result5["retriever"]["documents"], response_time5)
    
    # Scenario 6: Retail Banking - Mortgage Policy
    print_subsection("Scenario 6: Loan Officer - Mortgage Underwriting Standards")
    query6 = "What are the current mortgage underwriting standards and eligibility requirements?"
    filters6 = extract_filters_from_query(query6)
    
    start_time = datetime.now()
    result6 = search_pipeline.run({
        "text_embedder": {"text": query6},
        "retriever": {"filters": filters6}
    })
    response_time6 = (datetime.now() - start_time).total_seconds() * 1000
    
    display_search_results(query6, result6["retriever"]["documents"], filters6)
    log_query_audit("loan_officer_789", query6, filters6, 
                    result6["retriever"]["documents"], response_time6)
    
    # Scenario 7: Customer Service - Account Opening
    print_subsection("Scenario 7: Branch Staff - Account Opening Procedure")
    query7 = "What is the procedure for opening a new individual checking account?"
    filters7 = extract_filters_from_query(query7)
    
    start_time = datetime.now()
    result7 = search_pipeline.run({
        "text_embedder": {"text": query7},
        "retriever": {"filters": filters7}
    })
    response_time7 = (datetime.now() - start_time).total_seconds() * 1000
    
    display_search_results(query7, result7["retriever"]["documents"], filters7)
    log_query_audit("branch_staff_321", query7, filters7, 
                    result7["retriever"]["documents"], response_time7)
    
    # Scenario 8: Compliance - GDPR Requirements
    print_subsection("Scenario 8: Data Privacy Officer - GDPR Compliance")
    query8 = "What are the GDPR data protection requirements for customer information?"
    filters8 = extract_filters_from_query(query8)
    
    start_time = datetime.now()
    result8 = search_pipeline.run({
        "text_embedder": {"text": query8},
        "retriever": {"filters": filters8}
    })
    response_time8 = (datetime.now() - start_time).total_seconds() * 1000
    
    display_search_results(query8, result8["retriever"]["documents"], filters8)
    log_query_audit("privacy_officer_555", query8, filters8, 
                    result8["retriever"]["documents"], response_time8)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_section("SUMMARY")
    
    total_docs = document_store.count_documents()
    print(colored(f"\n✓ Total documents indexed: {total_docs}", Colors.GREEN))
    print(colored(f"✓ Search scenarios executed: 8", Colors.GREEN))
    print(colored(f"✓ Average response time: ~{(response_time1 + response_time2 + response_time3 + response_time4 + response_time5 + response_time6 + response_time7 + response_time8) / 8:.0f}ms", Colors.GREEN))
    print(colored(f"✓ All queries logged for audit compliance", Colors.GREEN))
    
    print(colored("\n" + "=" * 80, Colors.CYAN))
    print(colored("Banking Document Search Demo Complete!", Colors.BOLD + Colors.GREEN))
    print(colored("=" * 80, Colors.CYAN))
    
    print(colored("\nKey Features Demonstrated:", Colors.BLUE))
    print("  • Semantic vector search with DB2")
    print("  • Intelligent filter extraction from natural language")
    print("  • Multi-department document retrieval")
    print("  • Compliance audit logging")
    print("  • Fast query response times (<500ms)")
    print("  • Metadata-based filtering (type, department, date, regulation)")
    
    print(colored("\nNext Steps:", Colors.YELLOW))
    print("  1. Integrate with LLM for RAG (see banking_rag.py)")
    print("  2. Add user authentication and access control")
    print("  3. Implement audit trail database")
    print("  4. Deploy to production with monitoring")
    print("  5. Add multi-language support")


if __name__ == "__main__":
    main()

# Made with Bob
