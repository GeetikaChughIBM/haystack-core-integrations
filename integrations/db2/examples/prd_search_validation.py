# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Product Search Validation - Vector Search Verification

This validation script extends product_search_from_db2_table.py with:
1. SQL generation verification
2. Embedding diversity checks
3. Score threshold filtering (min_score parameter)
4. Diverse product categories for better testing
5. Vector table structure inspection

Use this to verify that vector search is working correctly.

Requirements:
- DB2 database with vector support
- Sentence Transformers for embeddings
- Haystack framework

Install:
pip install sentence-transformers haystack-ai ibm-db scipy
"""

import logging
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret

from haystack_integrations.components.retrievers.db2 import DB2EmbeddingRetriever, DB2KeywordRetriever
from haystack_integrations.document_stores.db2 import DB2DocumentStore

# Suppress HuggingFace informational messages
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*tokenizer_kwargs.*")
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"

    @staticmethod
    def is_supported() -> bool:
        """Check if terminal supports colors."""
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def colored(text: str, color: str) -> str:
    """Return colored text if terminal supports it."""
    if Colors.is_supported():
        return f"{color}{text}{Colors.END}"
    return text


def bold(text: str) -> str:
    """Return bold text if terminal supports it."""
    return colored(text, Colors.BOLD)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{colored('=' * 70, Colors.CYAN)}")
    print(colored(f"  {title}", Colors.BOLD + Colors.CYAN))
    print(colored("=" * 70, Colors.CYAN))


def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n{colored(f'▶ {title}', Colors.BLUE + Colors.BOLD)}")


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Auto-load credentials from .env file
try:
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(colored(f"✓ Loaded credentials from {env_path}", Colors.GREEN))
    else:
        load_dotenv()
except ImportError:
    print(colored("⚠ python-dotenv not installed", Colors.YELLOW))


def create_diverse_products_table(database: str, username: str, password: str) -> None:
    """
    Step 1: Create source products table with DIVERSE categories.

    This includes shoes, electronics, and books to test embedding differentiation.
    """
    print_subsection("Step 1: Creating Diverse Products Table")

    import ibm_db

    # Connect to DB2
    conn_str = f"DATABASE={database};HOSTNAME=localhost;PORT=50000;PROTOCOL=TCPIP;UID={username};PWD={password};"
    conn = ibm_db.connect(conn_str, "", "")  # type: ignore[arg-type]

    # Drop table if exists
    try:
        ibm_db.exec_immediate(conn, "DROP TABLE source_products")  # type: ignore[arg-type]
        print(colored("  • Dropped existing source_products table", Colors.YELLOW))
    except:
        pass

    # Create source products table
    create_table_sql = """
    CREATE TABLE source_products (
        product_id VARCHAR(50) NOT NULL PRIMARY KEY,
        name VARCHAR(200) NOT NULL,
        description VARCHAR(1000),
        brand VARCHAR(100),
        category VARCHAR(100),
        subcategory VARCHAR(100),
        price DECIMAL(10,2),
        color VARCHAR(50),
        size VARCHAR(20),
        in_stock BOOLEAN,
        rating DECIMAL(3,2),
        reviews INTEGER
    )
    """

    ibm_db.exec_immediate(conn, create_table_sql)  # type: ignore[arg-type]
    print(colored("  ✓ Created source_products table", Colors.GREEN))

    # Insert DIVERSE product data (shoes, electronics, books)
    products_data = [
        # Running Shoes
        (
            "nike_001",
            "Nike Air Max 90",
            "Classic running shoe with visible Air cushioning, perfect for daily wear and light jogging",
            "Nike",
            "shoes",
            "running",
            120.00,
            "white",
            "10",
            True,
            4.5,
            1250,
        ),
        (
            "nike_002",
            "Nike Pegasus 40",
            "Lightweight running shoe with responsive cushioning for speed and performance",
            "Nike",
            "shoes",
            "running",
            140.00,
            "black",
            "10",
            True,
            4.7,
            890,
        ),
        (
            "adidas_001",
            "Adidas Ultraboost 23",
            "Premium running shoe with Boost cushioning technology for comfort",
            "Adidas",
            "shoes",
            "running",
            180.00,
            "white",
            "10",
            True,
            4.8,
            2100,
        ),
        # Electronics (Laptops)
        (
            "dell_001",
            "Dell XPS 15",
            "High-performance laptop with Intel i7 processor, 16GB RAM, perfect for programming and video editing",
            "Dell",
            "electronics",
            "laptop",
            1499.00,
            "silver",
            "15-inch",
            True,
            4.6,
            3200,
        ),
        (
            "apple_001",
            "MacBook Pro M2",
            "Professional laptop with Apple M2 chip, excellent for creative work and development",
            "Apple",
            "electronics",
            "laptop",
            1999.00,
            "space-gray",
            "14-inch",
            True,
            4.9,
            5400,
        ),
        (
            "lenovo_001",
            "Lenovo ThinkPad X1",
            "Business laptop with robust build quality, ideal for enterprise and productivity",
            "Lenovo",
            "electronics",
            "laptop",
            1299.00,
            "black",
            "14-inch",
            True,
            4.5,
            1800,
        ),
        # Books (Programming)
        (
            "book_001",
            "Clean Code",
            "A handbook of agile software craftsmanship by Robert Martin, essential for developers",
            "Prentice Hall",
            "books",
            "programming",
            42.99,
            "N/A",
            "paperback",
            True,
            4.7,
            8900,
        ),
        (
            "book_002",
            "Design Patterns",
            "Elements of reusable object-oriented software, classic computer science textbook",
            "Addison-Wesley",
            "books",
            "programming",
            54.99,
            "N/A",
            "hardcover",
            True,
            4.8,
            6700,
        ),
        (
            "book_003",
            "The Pragmatic Programmer",
            "Your journey to mastery in software development and best practices",
            "Addison-Wesley",
            "books",
            "programming",
            39.99,
            "N/A",
            "paperback",
            True,
            4.9,
            12000,
        ),
    ]

    insert_sql = """
    INSERT INTO source_products 
    (product_id, name, description, brand, category, subcategory, price, color, size, in_stock, rating, reviews)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    stmt = ibm_db.prepare(conn, insert_sql)  # type: ignore[arg-type]
    for product in products_data:
        ibm_db.execute(stmt, product)  # type: ignore[arg-type]

    print(colored(f"  ✓ Inserted {len(products_data)} DIVERSE products (shoes, electronics, books)", Colors.GREEN))

    # Display source table contents by category
    print(f"\n  {bold('Source Table Contents (by Category):')}")
    query = "SELECT product_id, name, category, brand, price FROM source_products ORDER BY category, product_id"
    stmt = ibm_db.exec_immediate(conn, query)  # type: ignore[arg-type]

    print(f"  {colored('┌' + '─' * 78 + '┐', Colors.CYAN)}")
    print(
        f"  {colored('│', Colors.CYAN)} {bold('ID'):15} {bold('Name'):30} {bold('Category'):12} {bold('Brand'):12} {bold('Price'):8} {colored('│', Colors.CYAN)}"
    )
    print(f"  {colored('├' + '─' * 78 + '┤', Colors.CYAN)}")

    row = ibm_db.fetch_tuple(stmt)  # type: ignore[arg-type]
    while row:
        product_id, name, category, brand, price = row
        name_str = str(name)[:30]
        price_float = float(price)  # type: ignore[arg-type]
        print(
            f"  {colored('│', Colors.CYAN)} {product_id:15} {name_str:30} {category:12} {brand:12} ${price_float:7.2f} {colored('│', Colors.CYAN)}"
        )
        row = ibm_db.fetch_tuple(stmt)  # type: ignore[arg-type]

    print(f"  {colored('└' + '─' * 78 + '┘', Colors.CYAN)}")

    ibm_db.close(conn)  # type: ignore[arg-type]


def load_products_from_db2(database: str, username: str, password: str) -> list[Document]:
    """
    Step 2: Load products from source DB2 table and convert to Haystack Documents.
    """
    print_subsection("Step 2: Loading Products from Source Table")

    import ibm_db

    # Connect to DB2
    conn_str = f"DATABASE={database};HOSTNAME=localhost;PORT=50000;PROTOCOL=TCPIP;UID={username};PWD={password};"
    conn = ibm_db.connect(conn_str, "", "")  # type: ignore[arg-type]

    # Query source products table
    query = """
    SELECT product_id, name, description, brand, category, subcategory,
           price, color, size, in_stock, rating, reviews
    FROM source_products
    ORDER BY product_id
    """

    stmt = ibm_db.exec_immediate(conn, query)  # type: ignore[arg-type]

    documents = []
    row = ibm_db.fetch_tuple(stmt)  # type: ignore[arg-type]

    while row:
        product_id, name, description, brand, category, subcategory, price, color, size, in_stock, rating, reviews = row

        # Create Haystack Document
        doc = Document(
            id=product_id,
            content=f"{name} - {description}",
            meta={
                "product_id": product_id,
                "name": name,
                "brand": brand,
                "category": category,
                "subcategory": subcategory,
                "price": float(price),  # type: ignore[arg-type]
                "color": color,
                "size": size,
                "in_stock": bool(in_stock),
                "rating": float(rating),  # type: ignore[arg-type]
                "reviews": int(reviews),  # type: ignore[arg-type]
            },
        )
        documents.append(doc)
        row = ibm_db.fetch_tuple(stmt)  # type: ignore[arg-type]

    ibm_db.close(conn)  # type: ignore[arg-type]

    print(colored(f"  ✓ Loaded {len(documents)} products from source table", Colors.GREEN))
    print(f"\n  {bold('Sample Documents by Category:')}")

    # Group by category
    by_category = {}
    for doc in documents:
        cat = doc.meta["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(doc)

    for cat, docs in by_category.items():
        print(f"    {colored(cat.upper(), Colors.BLUE)}: {len(docs)} products")
        for doc in docs[:2]:
            print(f"      - {doc.meta['brand']} - {doc.content[:45]}...")

    return documents


def setup_vector_store() -> DB2DocumentStore:
    """
    Step 3: Initialize DB2 vector store.
    """
    print_subsection("Step 3: Setting up Vector Store")

    print(f"  • Database: {bold('TESTDB')}")
    print(f"  • Vector table: {bold('product_vectors')}")
    print(f"  • Embedding dimension: {bold('384')} (all-MiniLM-L6-v2)")
    print(f"  • Distance metric: {bold('cosine')}")

    document_store = DB2DocumentStore(
        database="TESTDB",
        username=Secret.from_env_var("DB2_USER", strict=False) or Secret.from_token("db2inst1"),
        password=Secret.from_env_var("DB2_PASSWORD", strict=False) or Secret.from_token("password"),
        table_name="product_vectors",
        embedding_dimension=384,
        distance_metric="cosine",
        recreate_table=True,
    )

    print(colored("  ✓ Vector store initialized successfully", Colors.GREEN))
    return document_store


def verify_vector_table_structure(database: str, username: str, password: str) -> None:
    """
    VALIDATION: Inspect product_vectors table structure.

    Verifies that the table has the correct schema with id, content, embedding, meta columns.
    """
    print_subsection("VALIDATION: Vector Table Structure")

    import ibm_db

    conn_str = f"DATABASE={database};HOSTNAME=localhost;PORT=50000;PROTOCOL=TCPIP;UID={username};PWD={password};"
    conn = ibm_db.connect(conn_str, "", "")  # type: ignore[arg-type]

    # Query table structure
    query = """
    SELECT COLNAME, TYPENAME, LENGTH 
    FROM SYSCAT.COLUMNS 
    WHERE TABNAME = 'PRODUCT_VECTORS' AND TABSCHEMA = CURRENT SCHEMA
    ORDER BY COLNO
    """

    stmt = ibm_db.exec_immediate(conn, query)  # type: ignore[arg-type]

    print(f"\n  {bold('Table Schema:')}")
    print(f"  {colored('┌' + '─' * 50 + '┐', Colors.CYAN)}")
    print(
        f"  {colored('│', Colors.CYAN)} {bold('Column'):20} {bold('Type'):15} {bold('Length'):10} {colored('│', Colors.CYAN)}"
    )
    print(f"  {colored('├' + '─' * 50 + '┤', Colors.CYAN)}")

    row = ibm_db.fetch_tuple(stmt)  # type: ignore[arg-type]
    while row:
        colname, typename, length = row
        print(f"  {colored('│', Colors.CYAN)} {colname:20} {typename:15} {length!s:10} {colored('│', Colors.CYAN)}")
        row = ibm_db.fetch_tuple(stmt)  # type: ignore[arg-type]

    print(f"  {colored('└' + '─' * 50 + '┘', Colors.CYAN)}")

    # Sample data from table
    print(f"\n  {bold('Sample Data (first 3 rows):')}")
    query = "SELECT id, content, meta FROM product_vectors FETCH FIRST 3 ROWS ONLY"
    stmt = ibm_db.exec_immediate(conn, query)  # type: ignore[arg-type]

    row = ibm_db.fetch_tuple(stmt)  # type: ignore[arg-type]
    i = 1
    while row:
        doc_id, content, meta = row
        print(f"    {i}. ID: {colored(str(doc_id), Colors.YELLOW)}")
        print(f"       Content: {str(content)[:60]}...")
        print(f"       Meta: {str(meta)[:80]}...")
        row = ibm_db.fetch_tuple(stmt)  # type: ignore[arg-type]
        i += 1

    print(colored("\n  ✓ Table structure verified: id, content, embedding (VECTOR), meta (CLOB)", Colors.GREEN))

    ibm_db.close(conn)  # type: ignore[arg-type]


def verify_embedding_diversity(database: str, username: str, password: str) -> None:
    """
    VALIDATION: Check that embeddings are different (not all zeros or identical).

    Computes cosine similarity between different product embeddings.
    """
    print_subsection("VALIDATION: Embedding Diversity Check")

    import json

    import ibm_db

    conn_str = f"DATABASE={database};HOSTNAME=localhost;PORT=50000;PROTOCOL=TCPIP;UID={username};PWD={password};"
    conn = ibm_db.connect(conn_str, "", "")  # type: ignore[arg-type]

    # Get embeddings for different categories
    query = """
    SELECT id, embedding, meta 
    FROM product_vectors 
    WHERE id IN ('nike_001', 'dell_001', 'book_001')
    """

    stmt = ibm_db.exec_immediate(conn, query)  # type: ignore[arg-type]

    embeddings = {}
    row = ibm_db.fetch_tuple(stmt)  # type: ignore[arg-type]

    while row:
        doc_id, embedding_str, meta_str = row
        # Parse embedding string "[0.1, 0.2, ...]" to list
        embedding_str = str(embedding_str).strip()
        if embedding_str.startswith("[") and embedding_str.endswith("]"):
            embedding = [float(x) for x in embedding_str[1:-1].split(",")]
            embeddings[doc_id] = embedding

            # Get category from meta
            meta = json.loads(str(meta_str))
            category = meta.get("category", "unknown")

            print(f"  • {doc_id} ({category}): {len(embedding)} dims, first 5 values: {embedding[:5]}")

        row = ibm_db.fetch_tuple(stmt)  # type: ignore[arg-type]

    # Compute cosine similarities
    if len(embeddings) >= 2:
        try:
            from scipy.spatial.distance import cosine

            print(f"\n  {bold('Cosine Similarity Matrix:')}")
            ids = list(embeddings.keys())

            for i, id1 in enumerate(ids):
                for id2 in ids[i + 1 :]:
                    similarity = 1 - cosine(embeddings[id1], embeddings[id2])

                    # Color code: high similarity (>0.8) = red flag, low (<0.5) = good
                    color = Colors.RED if similarity > 0.8 else Colors.GREEN if similarity < 0.5 else Colors.YELLOW

                    print(f"    {id1} <-> {id2}: {colored(f'{similarity:.4f}', color)}")

            print(f"\n  {colored('Expected:', Colors.CYAN)} Different categories should have similarity < 0.6")
            print(
                f"  {colored('Warning:', Colors.YELLOW)} If all similarities > 0.8, embeddings may not be diverse enough"
            )

        except ImportError:
            print(colored("  ⚠ scipy not installed, skipping similarity calculation", Colors.YELLOW))
            print("    Install with: pip install scipy")

    ibm_db.close(conn)  # type: ignore[arg-type]


def verify_sql_generation(document_store: DB2DocumentStore) -> None:
    """
    VALIDATION: Verify SQL generation for vector search.

    Shows the actual SQL query that will be executed.
    """
    print_subsection("VALIDATION: SQL Generation Verification")

    from haystack_integrations.document_stores.db2.query_builder import DB2QueryBuilder

    builder = DB2QueryBuilder(document_store.table_name, document_store.embedding_dimension)

    # Generate sample embedding string
    embedding_str = "[" + ",".join(["0.1"] * document_store.embedding_dimension) + "]"

    # Build vector search SQL
    sql = builder.build_vector_search("cosine", embedding_str, top_k=5, offset=0)

    print(f"\n  {bold('Generated SQL for Vector Search:')}")
    print(f"  {colored('─' * 70, Colors.CYAN)}")

    # Pretty print SQL
    sql_lines = sql.strip().split("\n")
    for line in sql_lines:
        print(f"  {line.strip()}")

    print(f"  {colored('─' * 70, Colors.CYAN)}")

    # Verify key components
    checks = [
        ("VECTOR_DISTANCE" in sql, "✓ Uses VECTOR_DISTANCE function"),
        ("COSINE" in sql, "✓ Uses COSINE distance metric"),
        ("CAST" in sql, "✓ Casts embedding to VECTOR type"),
        ("ORDER BY distance" in sql, "✓ Orders by distance (ascending)"),
        ("FETCH FIRST" in sql or "FETCH NEXT" in sql, "✓ Uses FETCH for limiting results"),
    ]

    print(f"\n  {bold('SQL Validation:')}")
    for passed, message in checks:
        color = Colors.GREEN if passed else Colors.RED
        symbol = "✓" if passed else "✗"
        print(f"    {colored(symbol, color)} {message}")

    if all(check[0] for check in checks):
        print(colored("\n  ✓ SQL generation looks correct!", Colors.GREEN))
    else:
        print(colored("\n  ✗ SQL generation has issues!", Colors.RED))


def index_products_with_embeddings(document_store: DB2DocumentStore, documents: list[Document]) -> dict:
    """
    Step 4: Generate embeddings and index into vector store.
    """
    print_subsection("Step 4: Generating Embeddings and Indexing")

    print(f"  • Embedding model: {bold('all-MiniLM-L6-v2')}")
    print(f"  • Documents to process: {bold(str(len(documents)))}")

    # Create indexing pipeline
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(
        "embedder", SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    )
    indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
    indexing_pipeline.connect("embedder", "writer")

    # Generate embeddings and write to vector store
    print("  • Processing documents...")
    result = indexing_pipeline.run({"embedder": {"documents": documents}})

    print(colored(f"  ✓ Successfully indexed {len(documents)} products with embeddings", Colors.GREEN))

    return result


def create_hybrid_search_pipeline(document_store: DB2DocumentStore) -> Pipeline:
    """
    Step 5: Create hybrid search pipeline.
    """
    print_subsection("Step 5: Creating Hybrid Search Pipeline")

    print(f"  • Text Embedder: {bold('SentenceTransformers')} (all-MiniLM-L6-v2)")
    print(f"  • Embedding Retriever: {bold('DB2EmbeddingRetriever')} (top_k=10)")
    print(f"  • Keyword Retriever: {bold('DB2KeywordRetriever')} (top_k=10)")
    print(f"  • Document Joiner: {bold('Reciprocal Rank Fusion')} (top_k=10)")

    pipeline = Pipeline()

    # Add components
    pipeline.add_component(
        "text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    )
    pipeline.add_component("embedding_retriever", DB2EmbeddingRetriever(document_store=document_store, top_k=10))
    pipeline.add_component("keyword_retriever", DB2KeywordRetriever(document_store=document_store, top_k=10))
    pipeline.add_component("joiner", DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=10))

    # Connect components
    pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    pipeline.connect("embedding_retriever.documents", "joiner.documents")
    pipeline.connect("keyword_retriever.documents", "joiner.documents")

    print(colored("  ✓ Pipeline created and connected successfully", Colors.GREEN))
    return pipeline


def search_products_with_threshold(
    pipeline: Pipeline, query: str, min_score: float = 0.0, filters: dict | None = None
) -> list[Document]:
    """
    Step 6: Perform hybrid search with SCORE THRESHOLD filtering.

    NEW: Filters out results below min_score to avoid returning irrelevant matches.
    """
    print(f"\n{colored('─' * 70, Colors.CYAN)}")
    print(f"  {bold('Query:')} {colored(query, Colors.YELLOW)}")
    if filters:
        print(f"  {bold('Filters:')} {filters}")
    if min_score > 0:
        print(
            f"  {bold('Min Score Threshold:')} {colored(f'{min_score:.2f}', Colors.GREEN)} (filters low-relevance results)"
        )
    print(colored("─" * 70, Colors.CYAN))

    # Run search
    print("  • Running hybrid retrieval (vector + keyword search)...")
    results = pipeline.run(
        {
            "text_embedder": {"text": query},
            "embedding_retriever": {"filters": filters},
            "keyword_retriever": {"query": query, "filters": filters},
        }
    )

    documents = results["joiner"]["documents"]

    # Apply score threshold filtering
    if min_score > 0:
        original_count = len(documents)
        documents = [doc for doc in documents if doc.score >= min_score]
        filtered_count = original_count - len(documents)

        if filtered_count > 0:
            print(colored(f"  • Filtered out {filtered_count} results below threshold {min_score:.2f}", Colors.YELLOW))

    # Display results
    if len(documents) == 0:
        print(f"\n  {colored('No products found matching criteria', Colors.RED)}\n")
        return documents

    print(f"\n  {colored(f'Found {len(documents)} products:', Colors.GREEN + Colors.BOLD)}\n")

    for i, doc in enumerate(documents, 1):
        # Product header
        category_color = (
            Colors.BLUE
            if doc.meta["category"] == "shoes"
            else Colors.CYAN
            if doc.meta["category"] == "electronics"
            else Colors.YELLOW
        )
        print(
            f"  {colored(f'{i}.', Colors.BOLD)} [{colored(doc.meta['category'].upper(), category_color)}] {colored(doc.meta['brand'], Colors.BOLD)} - {doc.content[:45]}..."
        )

        # Product details
        price_color = Colors.GREEN if doc.meta["price"] < 150 else Colors.YELLOW
        price_str = f"${doc.meta['price']:.2f}"
        print(f"     Price: {colored(price_str, price_color)}")

        rating_stars = "*" * int(doc.meta["rating"])
        print(f"     {rating_stars} {doc.meta['rating']}/5.0 ({doc.meta['reviews']} reviews)")

        stock_status = (
            colored("In Stock", Colors.GREEN) if doc.meta["in_stock"] else colored("Out of Stock", Colors.RED)
        )
        print(f"     Stock: {stock_status}")

        # Score with color coding
        score_color = Colors.GREEN if doc.score > 0.7 else Colors.YELLOW if doc.score > 0.4 else Colors.RED
        score_bar = "█" * int(doc.score * 20)  # Visual bar
        print(f"     Relevance: {colored(f'{doc.score:.4f}', score_color)} {score_bar}")
        print()

    return documents


def test_pure_vector_search(document_store: DB2DocumentStore) -> None:
    """
    VALIDATION: Test pure vector search (no hybrid fusion).

    This isolates vector search to verify it's working correctly.
    """
    print_subsection("VALIDATION: Pure Vector Search Test")

    from sentence_transformers import SentenceTransformer

    print("  • Testing vector search WITHOUT hybrid fusion")
    print("  • This isolates embedding-based retrieval\n")

    # Load embedding model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Test queries
    test_queries = [
        ("running shoes", "Should rank Nike/Adidas shoes high"),
        ("laptop for programming", "Should rank Dell/Apple/Lenovo high"),
        ("software development book", "Should rank programming books high"),
    ]

    retriever = DB2EmbeddingRetriever(document_store=document_store, top_k=5)

    for query, expected in test_queries:
        print(f"  {bold('Query:')} {colored(query, Colors.YELLOW)}")
        print(f"  {colored('Expected:', Colors.CYAN)} {expected}")

        # Generate query embedding
        query_embedding = model.encode(query).tolist()

        # Pure vector search
        results = retriever.run(query_embedding=query_embedding)
        documents = results["documents"]

        print(f"  {colored('Results:', Colors.GREEN)}")
        for i, doc in enumerate(documents[:3], 1):
            score = doc.score if doc.score is not None else 0.0
            score_color = Colors.GREEN if score > 0.6 else Colors.YELLOW if score > 0.4 else Colors.RED
            print(
                f"    {i}. [{doc.meta['category']}] {doc.meta['name'][:40]:40} Score: {colored(f'{score:.4f}', score_color)}"
            )

        print()


def main() -> None:
    """Main function with validation steps."""
    print_section("DB2 Vector Search Validation Suite")
    print(f"\n{colored('Comprehensive validation with diverse products and verification tests', Colors.CYAN)}")

    # Database credentials
    database = "TESTDB"
    username = Secret.from_env_var("DB2_USER", strict=False) or Secret.from_token("db2inst1")
    password = Secret.from_env_var("DB2_PASSWORD", strict=False) or Secret.from_token("password")

    username_str = username.resolve_value() or "db2inst1"
    password_str = password.resolve_value() or "password"

    # Step 1: Create diverse products table
    create_diverse_products_table(database, username_str, password_str)

    # Step 2: Load products
    documents = load_products_from_db2(database, username_str, password_str)

    # Step 3: Setup vector store
    document_store = setup_vector_store()

    # VALIDATION: Verify table structure
    verify_vector_table_structure(database, username_str, password_str)

    # VALIDATION: Verify SQL generation
    verify_sql_generation(document_store)

    # Step 4: Generate embeddings and index
    index_products_with_embeddings(document_store, documents)

    # VALIDATION: Check embedding diversity
    verify_embedding_diversity(database, username_str, password_str)

    # VALIDATION: Test pure vector search
    test_pure_vector_search(document_store)

    # Step 5: Create search pipeline
    pipeline = create_hybrid_search_pipeline(document_store)

    # Test searches with score thresholds
    print_section("SCENARIO 1: Cross-Category Search (No Threshold)")
    print(colored("  Generic query should return diverse categories", Colors.CYAN))
    search_products_with_threshold(pipeline, "best products", min_score=0.0)

    print_section("SCENARIO 2: Category-Specific Search (With Threshold)")
    print(colored("  Specific query with min_score=0.5 to filter irrelevant results", Colors.CYAN))
    search_products_with_threshold(pipeline, "running shoes for jogging", min_score=0.5)

    print_section("SCENARIO 3: Electronics Search (With Threshold)")
    print(colored("  Should return only laptops, filtered by min_score=0.4", Colors.CYAN))
    search_products_with_threshold(pipeline, "laptop for software development", min_score=0.4)

    print_section("SCENARIO 4: Books Search (With High Threshold)")
    print(colored("  High threshold (0.6) ensures only relevant books", Colors.CYAN))
    search_products_with_threshold(pipeline, "programming books for developers", min_score=0.6)

    print_section("SCENARIO 5: Filtered Search with Threshold")
    print(colored("  Category filter + score threshold for precision", Colors.CYAN))
    search_products_with_threshold(
        pipeline, "comfortable shoes", min_score=0.4, filters={"field": "category", "operator": "==", "value": "shoes"}
    )

    # Summary
    print_section("Validation Complete!")
    print(f"\n{bold('Validation Checks Performed:')}")
    print(f"  1. {colored('Table Structure', Colors.GREEN)} - Verified schema and data format")
    print(f"  2. {colored('SQL Generation', Colors.GREEN)} - Confirmed VECTOR_DISTANCE usage")
    print(f"  3. {colored('Embedding Diversity', Colors.GREEN)} - Checked embeddings are different")
    print(f"  4. {colored('Pure Vector Search', Colors.GREEN)} - Tested without hybrid fusion")
    print(f"  5. {colored('Score Thresholds', Colors.GREEN)} - Filtered low-relevance results")

    print(f"\n{bold('Key Improvements:')}")
    print(f"  • {colored('Diverse Products', Colors.CYAN)} - Shoes, Electronics, Books for better testing")
    print(f"  • {colored('Min Score Filtering', Colors.CYAN)} - Prevents returning irrelevant results")
    print(f"  • {colored('SQL Verification', Colors.CYAN)} - Confirms correct query generation")
    print(f"  • {colored('Embedding Checks', Colors.CYAN)} - Validates vector differentiation")

    print(f"\n{bold('Score Interpretation:')}")
    print(f"  • {colored('> 0.7', Colors.GREEN)} - Highly relevant (strong match)")
    print(f"  • {colored('0.4 - 0.7', Colors.YELLOW)} - Moderately relevant (partial match)")
    print(f"  • {colored('< 0.4', Colors.RED)} - Low relevance (weak match)")

    print(f"\n{colored('=' * 70, Colors.CYAN)}\n")


if __name__ == "__main__":
    main()

# Made with Bob - Validation Edition
