# Banking Use Case: Intelligent Document Search & RAG with DB2 + Haystack

## Overview

This banking use case demonstrates how financial institutions can leverage DB2's vector capabilities with Haystack to build intelligent document search and RAG (Retrieval-Augmented Generation) systems for:

- **Regulatory Compliance**: Search across policies, regulations, and compliance documents
- **Customer Support**: Intelligent FAQ and policy retrieval for customer service agents
- **Risk Management**: Search financial reports, risk assessments, and audit documents
- **Product Information**: Retrieve loan terms, credit card policies, investment products
- **Internal Knowledge Base**: Employee handbook, procedures, training materials

---

## Business Value

### 1. **Compliance & Regulatory**
- ✅ Instant access to relevant regulations (Basel III, Dodd-Frank, GDPR, etc.)
- ✅ Semantic search across thousands of compliance documents
- ✅ Audit trail with document versioning
- ✅ Multi-language support for global operations

### 2. **Customer Service Excellence**
- ✅ AI-powered chatbots with accurate policy information
- ✅ Reduce average handling time by 40%
- ✅ Consistent answers across all channels
- ✅ Real-time policy updates

### 3. **Risk Management**
- ✅ Quick access to risk assessments and mitigation strategies
- ✅ Historical analysis of similar risk scenarios
- ✅ Regulatory change impact analysis
- ✅ Fraud detection pattern matching

### 4. **Operational Efficiency**
- ✅ Reduce manual document search time by 80%
- ✅ Automated document classification
- ✅ Intelligent routing of customer inquiries
- ✅ Knowledge retention and transfer

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Banking Application Layer                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Web UI     │  │  Mobile App  │  │   Chatbot    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼──────────────────┼──────────────────┼──────────────────┘
          │                  │                  │
          └──────────────────┴──────────────────┘
                             │
┌────────────────────────────▼──────────────────────────────────────┐
│                    Haystack RAG Pipeline                           │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              Query Processing & Routing                     │  │
│  │  • Intent Classification                                    │  │
│  │  • Query Enhancement                                        │  │
│  │  • Filter Extraction (department, date, type)              │  │
│  └────────────────────────┬───────────────────────────────────┘  │
│                            │                                       │
│  ┌────────────────────────▼───────────────────────────────────┐  │
│  │              Document Retrieval Layer                       │  │
│  │                                                             │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │  │
│  │  │   Vector     │  │   Keyword    │  │   Hybrid     │    │  │
│  │  │   Search     │  │   Search     │  │   Search     │    │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │  │
│  │         │                  │                  │             │  │
│  │         └──────────────────┴──────────────────┘             │  │
│  │                            │                                 │  │
│  │  ┌────────────────────────▼───────────────────────────┐    │  │
│  │  │         DB2 Document Store (Vector)                 │    │  │
│  │  │  • Embedding Retriever                              │    │  │
│  │  │  • Metadata Filtering (dept, date, type, status)    │    │  │
│  │  │  • Pagination & Ranking                             │    │  │
│  │  └─────────────────────────────────────────────────────┘    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            │                                       │
│  ┌────────────────────────▼───────────────────────────────────┐  │
│  │              Answer Generation (RAG)                        │  │
│  │  • Context Assembly                                         │  │
│  │  • LLM (GPT-4, Claude, Llama)                              │  │
│  │  • Citation & Source Tracking                               │  │
│  │  • Compliance Validation                                    │  │
│  └─────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────▼──────────────────────────────────────┐
│                      IBM DB2 Database                              │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              Banking Documents Table                        │  │
│  │  • id (VARCHAR PRIMARY KEY)                                 │  │
│  │  • content (CLOB) - Document text                           │  │
│  │  • embedding (VECTOR) - 384/768/1536 dimensions             │  │
│  │  • meta (CLOB) - JSON metadata:                             │  │
│  │    - document_type (policy, regulation, faq, etc.)          │  │
│  │    - department (compliance, risk, retail, etc.)            │  │
│  │    - effective_date (YYYY-MM-DD)                            │  │
│  │    - expiry_date (YYYY-MM-DD)                               │  │
│  │    - status (active, archived, draft)                       │  │
│  │    - regulation_id (Basel III, Dodd-Frank, etc.)            │  │
│  │    - classification (public, internal, confidential)        │  │
│  │    - version (1.0, 2.0, etc.)                               │  │
│  │    - language (en, es, fr, de)                              │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              Audit Trail Table                              │  │
│  │  • query_id (VARCHAR PRIMARY KEY)                           │  │
│  │  • user_id (VARCHAR)                                        │  │
│  │  • query_text (VARCHAR)                                     │  │
│  │  • retrieved_docs (CLOB) - JSON array of doc IDs            │  │
│  │  • timestamp (TIMESTAMP)                                    │  │
│  │  • department (VARCHAR)                                     │  │
│  └────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

---

## Document Types & Metadata Schema

### 1. **Regulatory Documents**
```json
{
  "document_type": "regulation",
  "department": "compliance",
  "regulation_id": "Basel-III-2023",
  "title": "Basel III Capital Requirements",
  "effective_date": "2023-01-01",
  "expiry_date": null,
  "status": "active",
  "classification": "internal",
  "version": "3.1",
  "language": "en",
  "jurisdiction": "US",
  "tags": ["capital-adequacy", "risk-management", "tier-1-capital"]
}
```

### 2. **Customer FAQs**
```json
{
  "document_type": "faq",
  "department": "customer_service",
  "category": "credit_cards",
  "subcategory": "rewards_program",
  "effective_date": "2024-01-01",
  "status": "active",
  "classification": "public",
  "version": "2.0",
  "language": "en",
  "tags": ["rewards", "points", "redemption"]
}
```

### 3. **Risk Assessments**
```json
{
  "document_type": "risk_assessment",
  "department": "risk_management",
  "risk_category": "credit_risk",
  "severity": "high",
  "assessment_date": "2024-03-15",
  "status": "active",
  "classification": "confidential",
  "version": "1.0",
  "language": "en",
  "tags": ["credit-default", "portfolio-risk", "mitigation"]
}
```

### 4. **Product Policies**
```json
{
  "document_type": "policy",
  "department": "retail_banking",
  "product_type": "mortgage",
  "policy_id": "MTG-2024-001",
  "effective_date": "2024-01-01",
  "expiry_date": "2024-12-31",
  "status": "active",
  "classification": "internal",
  "version": "5.2",
  "language": "en",
  "tags": ["interest-rates", "eligibility", "documentation"]
}
```

### 5. **Internal Procedures**
```json
{
  "document_type": "procedure",
  "department": "operations",
  "procedure_id": "KYC-2024-001",
  "title": "Know Your Customer (KYC) Verification",
  "effective_date": "2024-01-01",
  "status": "active",
  "classification": "internal",
  "version": "3.0",
  "language": "en",
  "tags": ["kyc", "aml", "customer-onboarding"]
}
```

---

## Use Case Scenarios

### Scenario 1: Compliance Officer - Regulatory Search

**User Query**: "What are the capital adequacy requirements under Basel III for Tier 1 capital?"

**System Flow**:
1. **Query Processing**: Extract intent (regulatory search) and filters (Basel III, Tier 1 capital)
2. **Retrieval**: Vector search with filters:
   ```python
   filters = {
       "document_type": "regulation",
       "regulation_id": {"$in": ["Basel-III-2023", "Basel-III-2022"]},
       "status": "active",
       "tags": {"$in": ["tier-1-capital", "capital-adequacy"]}
   }
   ```
3. **Results**: Top 5 relevant regulation sections
4. **RAG Generation**: LLM synthesizes answer with citations
5. **Audit**: Log query, user, retrieved documents, timestamp

**Expected Output**:
```
Based on Basel III regulations (effective 2023), Tier 1 capital requirements are:

1. Common Equity Tier 1 (CET1): Minimum 4.5% of risk-weighted assets
2. Additional Tier 1 (AT1): Minimum 1.5% of risk-weighted assets
3. Total Tier 1: Minimum 6.0% of risk-weighted assets

Additionally, banks must maintain a capital conservation buffer of 2.5%.

Sources:
- Basel III Capital Requirements (Version 3.1, Section 4.2)
- Capital Adequacy Framework (Version 2.0, Section 2.1)
```

---

### Scenario 2: Customer Service Agent - Product Information

**User Query**: "Customer asking about mortgage refinancing options and current interest rates"

**System Flow**:
1. **Query Processing**: Extract intent (product inquiry) and category (mortgage)
2. **Retrieval**: Hybrid search (vector + keyword) with filters:
   ```python
   filters = {
       "document_type": {"$in": ["policy", "faq"]},
       "product_type": "mortgage",
       "status": "active",
       "classification": {"$in": ["public", "internal"]}
   }
   ```
3. **Results**: Top 10 relevant documents (policies + FAQs)
4. **RAG Generation**: LLM creates customer-friendly response
5. **Audit**: Log for quality assurance

**Expected Output**:
```
Mortgage Refinancing Options:

1. Rate-and-Term Refinance:
   - Current rates: 6.25% - 6.75% APR (30-year fixed)
   - Eligibility: Credit score 620+, LTV ratio ≤ 80%
   - Processing time: 30-45 days

2. Cash-Out Refinance:
   - Current rates: 6.50% - 7.00% APR (30-year fixed)
   - Eligibility: Credit score 640+, LTV ratio ≤ 75%
   - Maximum cash-out: 80% of home value

Required Documents:
- Recent pay stubs (2 months)
- Tax returns (2 years)
- Current mortgage statement
- Home appraisal

Sources:
- Mortgage Refinancing Policy (MTG-2024-001, Section 3)
- Current Interest Rates (Updated: 2024-05-01)
```

---

### Scenario 3: Risk Manager - Historical Risk Analysis

**User Query**: "Show me similar credit risk scenarios from the past 2 years related to commercial real estate"

**System Flow**:
1. **Query Processing**: Extract intent (risk analysis) and filters (credit risk, CRE, date range)
2. **Retrieval**: Vector search with temporal filters:
   ```python
   filters = {
       "document_type": "risk_assessment",
       "risk_category": "credit_risk",
       "assessment_date": {"$gte": "2022-05-01"},
       "tags": {"$in": ["commercial-real-estate", "cre", "property-risk"]},
       "status": {"$in": ["active", "archived"]}
   }
   ```
3. **Results**: Top 15 historical risk assessments
4. **RAG Generation**: LLM summarizes patterns and mitigation strategies
5. **Audit**: Log for risk management review

**Expected Output**:
```
Historical Credit Risk Analysis - Commercial Real Estate (2022-2024):

Key Findings:
1. 12 high-severity risk assessments identified
2. Common risk factors:
   - Rising interest rates (8 cases)
   - Occupancy rate decline (6 cases)
   - Market oversupply (4 cases)

Mitigation Strategies Applied:
1. Enhanced due diligence for new CRE loans
2. Stress testing at 200 bps rate increase
3. Quarterly portfolio reviews
4. Loan-to-value ratio caps at 65%

Outcomes:
- Default rate: 2.3% (below industry average of 3.1%)
- Average recovery rate: 78%
- Portfolio risk rating improved from B+ to A-

Sources:
- CRE Risk Assessment Q1-2024 (RA-2024-015)
- CRE Risk Assessment Q4-2023 (RA-2023-048)
- CRE Risk Assessment Q3-2023 (RA-2023-037)
[+ 9 more assessments]
```

---

### Scenario 4: Fraud Analyst - Pattern Matching

**User Query**: "Find fraud cases involving wire transfer requests with similar patterns to case #FR-2024-0156"

**System Flow**:
1. **Query Processing**: Load reference case embedding
2. **Retrieval**: Vector similarity search:
   ```python
   filters = {
       "document_type": "fraud_case",
       "fraud_type": "wire_transfer",
       "status": {"$in": ["confirmed", "under_investigation"]},
       "case_date": {"$gte": "2023-01-01"}
   }
   ```
3. **Results**: Top 20 similar fraud cases
4. **RAG Generation**: LLM identifies common patterns
5. **Audit**: Log for fraud investigation tracking

**Expected Output**:
```
Similar Fraud Cases - Wire Transfer Pattern Analysis:

Reference Case: FR-2024-0156
- Amount: $45,000
- Method: Business email compromise
- Target: Small business account

Similar Cases Found: 18 cases (similarity > 0.85)

Common Patterns:
1. Email Spoofing (16/18 cases):
   - Impersonation of vendor/supplier
   - Urgent payment requests
   - Slight domain variations

2. Timing (14/18 cases):
   - Requests sent Friday afternoon
   - Payment deadline: Monday morning
   - Outside normal business hours

3. Account Characteristics (12/18 cases):
   - Business accounts < 2 years old
   - First-time large wire transfer
   - Destination: Foreign account

Prevention Measures:
- Multi-factor authentication for wire transfers > $10,000
- Callback verification for new payees
- 24-hour hold for first-time large transfers
- Enhanced monitoring for Friday afternoon requests

Sources:
- Fraud Case FR-2024-0142 (Similarity: 0.92)
- Fraud Case FR-2024-0128 (Similarity: 0.89)
- Fraud Case FR-2023-0891 (Similarity: 0.87)
[+ 15 more cases]
```

---

## Technical Implementation

### 1. **Document Ingestion Pipeline**

```python
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.db2 import DB2DocumentStore

# Initialize document store
document_store = DB2DocumentStore(
    database="BANKDB",
    username=Secret.from_env_var("DB2_USER"),
    password=Secret.from_env_var("DB2_PASSWORD"),
    table_name="banking_documents",
    embedding_dimension=384,
    distance_metric="cosine",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    validate_embedding_model=True
)

# Create ingestion pipeline
ingestion_pipeline = Pipeline()
ingestion_pipeline.add_component(
    "embedder",
    SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
)
ingestion_pipeline.add_component(
    "writer",
    DocumentWriter(document_store=document_store)
)
ingestion_pipeline.connect("embedder.documents", "writer.documents")

# Ingest documents
documents = load_banking_documents()  # Load from CSV/JSON/Database
ingestion_pipeline.run({"embedder": {"documents": documents}})
```

### 2. **RAG Pipeline with LLM**

```python
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.retrievers.db2 import DB2EmbeddingRetriever

# Create RAG pipeline
rag_pipeline = Pipeline()

# Add components
rag_pipeline.add_component(
    "text_embedder",
    SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
)
rag_pipeline.add_component(
    "retriever",
    DB2EmbeddingRetriever(
        document_store=document_store,
        top_k=10
    )
)
rag_pipeline.add_component(
    "prompt_builder",
    PromptBuilder(template=banking_prompt_template)
)
rag_pipeline.add_component(
    "llm",
    OpenAIGenerator(model="gpt-4", api_key=Secret.from_env_var("OPENAI_API_KEY"))
)

# Connect components
rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "llm.prompt")

# Run query
result = rag_pipeline.run({
    "text_embedder": {"text": "What are Basel III capital requirements?"},
    "retriever": {
        "filters": {
            "document_type": "regulation",
            "status": "active"
        }
    }
})
```

### 3. **Prompt Template for Banking**

```python
banking_prompt_template = """
You are a banking compliance and policy expert assistant. Answer the user's question based ONLY on the provided documents.

Rules:
1. Provide accurate, factual information from the documents
2. Include document citations (title, version, section)
3. If information is not in the documents, say "I don't have information about that"
4. Use clear, professional language
5. For regulatory questions, emphasize compliance requirements
6. For customer questions, provide actionable steps

Documents:
{% for doc in documents %}
---
Document: {{ doc.meta.title }}
Type: {{ doc.meta.document_type }}
Version: {{ doc.meta.version }}
Content: {{ doc.content }}
---
{% endfor %}

User Question: {{ question }}

Answer:
"""
```

### 4. **Intelligent Filter Extraction**

```python
import re
from datetime import datetime, timedelta

def extract_filters_from_query(query: str) -> dict:
    """Extract metadata filters from natural language query."""
    filters = {}
    
    # Document type detection
    if any(word in query.lower() for word in ["regulation", "regulatory", "compliance"]):
        filters["document_type"] = "regulation"
    elif any(word in query.lower() for word in ["policy", "procedure"]):
        filters["document_type"] = {"$in": ["policy", "procedure"]}
    elif any(word in query.lower() for word in ["faq", "question", "customer"]):
        filters["document_type"] = "faq"
    elif any(word in query.lower() for word in ["risk", "assessment"]):
        filters["document_type"] = "risk_assessment"
    
    # Department detection
    if "compliance" in query.lower():
        filters["department"] = "compliance"
    elif any(word in query.lower() for word in ["risk", "credit risk", "market risk"]):
        filters["department"] = "risk_management"
    elif any(word in query.lower() for word in ["retail", "consumer", "customer"]):
        filters["department"] = "retail_banking"
    
    # Date range detection
    if "last year" in query.lower():
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        filters["effective_date"] = {"$gte": one_year_ago}
    elif "last 2 years" in query.lower():
        two_years_ago = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        filters["effective_date"] = {"$gte": two_years_ago}
    
    # Regulation ID detection
    basel_match = re.search(r"basel\s*(iii|3)", query.lower())
    if basel_match:
        filters["regulation_id"] = {"$in": ["Basel-III-2023", "Basel-III-2022"]}
    
    if "dodd-frank" in query.lower():
        filters["regulation_id"] = {"$in": ["Dodd-Frank-2023", "Dodd-Frank-2022"]}
    
    # Status filter (always active unless specified)
    if "archived" not in query.lower():
        filters["status"] = "active"
    
    return filters
```

---

## Security & Compliance Features

### 1. **Access Control**
```python
def check_user_access(user_id: str, document: Document) -> bool:
    """Verify user has access to document based on classification."""
    user_clearance = get_user_clearance(user_id)
    doc_classification = document.meta.get("classification", "public")
    
    clearance_levels = {
        "public": 0,
        "internal": 1,
        "confidential": 2,
        "restricted": 3
    }
    
    return clearance_levels[user_clearance] >= clearance_levels[doc_classification]
```

### 2. **Audit Logging**
```python
def log_query_audit(user_id: str, query: str, retrieved_docs: List[Document]):
    """Log all queries for compliance audit trail."""
    audit_entry = {
        "query_id": generate_uuid(),
        "user_id": user_id,
        "query_text": query,
        "retrieved_docs": [doc.id for doc in retrieved_docs],
        "timestamp": datetime.now().isoformat(),
        "department": get_user_department(user_id)
    }
    
    # Store in audit table
    store_audit_log(audit_entry)
```

### 3. **Data Masking**
```python
def mask_sensitive_data(content: str) -> str:
    """Mask PII and sensitive information in responses."""
    # Mask SSN
    content = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', 'XXX-XX-XXXX', content)
    
    # Mask account numbers
    content = re.sub(r'\b\d{10,16}\b', 'XXXX-XXXX-XXXX', content)
    
    # Mask email addresses
    content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                     'user@example.com', content)
    
    return content
```

---

## Performance Metrics

### Expected Performance
- **Query Latency**: < 500ms (vector search + retrieval)
- **RAG Response Time**: < 3 seconds (including LLM generation)
- **Throughput**: 100+ queries/second
- **Accuracy**: 95%+ relevant document retrieval
- **Uptime**: 99.9% availability

### Monitoring
```python
from prometheus_client import Counter, Histogram

# Metrics
query_counter = Counter('banking_queries_total', 'Total queries', ['department', 'document_type'])
query_latency = Histogram('banking_query_latency_seconds', 'Query latency')
retrieval_accuracy = Histogram('banking_retrieval_accuracy', 'Retrieval accuracy score')

# Track metrics
with query_latency.time():
    results = rag_pipeline.run(query_data)
    
query_counter.labels(
    department=filters.get("department", "unknown"),
    document_type=filters.get("document_type", "unknown")
).inc()
```

---

## Cost Analysis

### Infrastructure Costs (Monthly)
- **DB2 Database**: $500-2000 (depending on size and tier)
- **Embedding Model**: $0 (open-source Sentence Transformers)
- **LLM API (GPT-4)**: $1000-5000 (based on usage)
- **Compute**: $500-1500 (application servers)
- **Total**: $2000-9000/month

### ROI Calculation
- **Time Savings**: 80% reduction in document search time
  - Average search time: 15 min → 2 min (13 min saved)
  - 100 searches/day × 13 min = 1,300 min/day = 21.7 hours/day
  - Annual savings: ~5,400 hours × $50/hour = $270,000

- **Compliance Risk Reduction**: $500,000+ (avoiding regulatory fines)
- **Customer Satisfaction**: 40% reduction in handling time = $100,000+ savings
- **Total Annual ROI**: $870,000+ savings vs $24,000-108,000 cost

**Payback Period**: 1-2 months

---

## Next Steps

1. **Phase 1: Proof of Concept (2-4 weeks)**
   - Set up DB2 document store
   - Ingest 1,000 sample documents
   - Build basic RAG pipeline
   - Test with 10 use cases

2. **Phase 2: Pilot (1-2 months)**
   - Expand to 10,000 documents
   - Integrate with existing systems
   - User acceptance testing
   - Performance optimization

3. **Phase 3: Production (2-3 months)**
   - Full document corpus (100,000+ documents)
   - Multi-language support
   - Advanced security features
   - Monitoring and alerting

4. **Phase 4: Enhancement (Ongoing)**
   - Fine-tune embedding models
   - Add more document types
   - Integrate with additional data sources
   - Continuous improvement based on feedback

---

## Conclusion

This banking use case demonstrates the power of combining DB2's enterprise-grade vector capabilities with Haystack's flexible RAG framework to build intelligent document search and question-answering systems for financial institutions.

**Key Benefits**:
- ✅ Enterprise-grade security and compliance
- ✅ Scalable to millions of documents
- ✅ Fast, accurate semantic search
- ✅ Audit trail for regulatory compliance
- ✅ Multi-language support
- ✅ Integration with existing DB2 infrastructure
- ✅ Significant ROI and time savings

**Ready to get started?** See the implementation examples in this directory!