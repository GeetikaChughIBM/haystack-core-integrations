# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Product Search from Existing DB2 Table

This example demonstrates a realistic production workflow:
1. Create a source DB2 table with product data (simulating existing business data)
2. Load products from the DB2 table
3. Generate embeddings and index into vector store
4. Perform hybrid search (vector + keyword) with filters

This approach is ideal when you have existing product data in DB2 and want to
add semantic search capabilities without modifying your source tables.

Requirements:
- DB2 database with vector support
- Sentence Transformers for embeddings
- Haystack framework

Install:
pip install sentence-transformers haystack-ai ibm-db
"""

import logging
import re
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


def create_source_products_table(database: str, username: str, password: str) -> None:
    """
    Step 1: Create source products table (simulating existing business data).

    In production, this table would already exist with your product catalog.
    """
    print_subsection("Step 1: Creating Source Products Table")

    import ibm_db

    # Connect to DB2
    conn_str = f"DATABASE={database};HOSTNAME=localhost;PORT=50000;PROTOCOL=TCPIP;UID={username};PWD={password};"
    conn = ibm_db.connect(conn_str, "", "")   

    # Drop table if exists
    try:
        ibm_db.exec_immediate(conn, "DROP TABLE source_products")   
        print(colored("  • Dropped existing source_products table", Colors.YELLOW))
    except:
        pass

    # Create source products table (traditional relational table)
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

    ibm_db.exec_immediate(conn, create_table_sql)   
    print(colored("  ✓ Created source_products table", Colors.GREEN))

    # Insert sample product data with RICH DESCRIPTIONS
    products_data = [
        (
            "nike_001",
            "Nike Air Max 90",
            "Nike Air Max 90 is a classic white running shoe priced at $120. This Nike brand shoe features visible Air cushioning technology, perfect for daily wear and light jogging. Available in size 10, currently in stock. Highly rated at 4.5 stars with 1250 customer reviews. Ideal for runners seeking comfort and style in the running shoes category.",
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
            "Nike Pegasus 40 is a lightweight black running shoe priced at $140. This Nike brand performance shoe offers responsive cushioning designed for speed and performance. Available in size 10, currently in stock. Excellent rating of 4.7 stars with 890 customer reviews. Perfect for serious runners in the running shoes category.",
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
            "nike_003",
            "Nike ZoomX Vaporfly",
            "Nike ZoomX Vaporfly is a professional pink racing shoe priced at $250. This premium Nike brand shoe features carbon fiber plate technology for maximum speed. Available in size 10, currently out of stock. Outstanding rating of 4.9 stars with 450 customer reviews. Elite choice for competitive runners in the racing shoes category.",
            "Nike",
            "shoes",
            "racing",
            250.00,
            "pink",
            "10",
            False,
            4.9,
            450,
        ),
        (
            "adidas_001",
            "Adidas Ultraboost 23",
            "Adidas Ultraboost 23 is a premium white running shoe priced at $180. This Adidas brand shoe features revolutionary Boost cushioning technology for exceptional comfort. Available in size 10, currently in stock. Highly rated at 4.8 stars with 2100 customer reviews. Top choice for comfort-focused runners in the running shoes category.",
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
        (
            "adidas_002",
            "Adidas Adizero Boston",
            "Adidas Adizero Boston is a lightweight blue training shoe priced at $140. This Adidas brand shoe is designed for tempo runs and speed work. Available in size 10, currently in stock. Rated 4.6 stars with 670 customer reviews. Excellent for speed training in the running shoes category.",
            "Adidas",
            "shoes",
            "running",
            140.00,
            "blue",
            "10",
            True,
            4.6,
            670,
        ),
        (
            "nike_004",
            "Nike React Infinity Run",
            "Nike React Infinity Run is a stability gray running shoe priced at $160. This Nike brand shoe is specifically designed to reduce injury risk with advanced stability features. Available in size 10, currently in stock. Rated 4.4 stars with 980 customer reviews. Great for injury-prone runners in the running shoes category.",
            "Nike",
            "shoes",
            "running",
            160.00,
            "gray",
            "10",
            True,
            4.4,
            980,
        ),
    ]

    insert_sql = """
    INSERT INTO source_products 
    (product_id, name, description, brand, category, subcategory, price, color, size, in_stock, rating, reviews)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    stmt = ibm_db.prepare(conn, insert_sql)   
    for product in products_data:
        ibm_db.execute(stmt, product)   

    print(colored(f"  ✓ Inserted {len(products_data)} products into source table", Colors.GREEN))

    # Display source table contents
    print(f"\n  {bold('Source Table Contents:')}")
    query = "SELECT product_id, name, brand, price, in_stock FROM source_products ORDER BY product_id"
    stmt = ibm_db.exec_immediate(conn, query)   

    print(f"  {colored('┌' + '─' * 66 + '┐', Colors.CYAN)}")
    print(
        f"  {colored('│', Colors.CYAN)} {bold('ID'):15} {bold('Name'):25} {bold('Brand'):10} {bold('Price'):8} {colored('│', Colors.CYAN)}"
    )
    print(f"  {colored('├' + '─' * 66 + '┤', Colors.CYAN)}")

    row = ibm_db.fetch_tuple(stmt)   
    while row:
        product_id, name, brand, price, in_stock = row
        stock_icon = "✓" if in_stock else "✗"
        name_str = str(name)[:25]
        price_float = float(price)   
        print(
            f"  {colored('│', Colors.CYAN)} {product_id:15} {name_str:25} {brand:10} ${price_float:6.2f} {colored('│', Colors.CYAN)}"
        )
        row = ibm_db.fetch_tuple(stmt)   

    print(f"  {colored('└' + '─' * 66 + '┘', Colors.CYAN)}")

    ibm_db.close(conn)   


def load_products_from_db2(database: str, username: str, password: str) -> list[Document]:
    """
    Step 2: Load products from source DB2 table and convert to Haystack Documents.

    This simulates reading from your existing product catalog.
    """
    print_subsection("Step 2: Loading Products from Source Table")

    import ibm_db

    # Connect to DB2
    conn_str = f"DATABASE={database};HOSTNAME=localhost;PORT=50000;PROTOCOL=TCPIP;UID={username};PWD={password};"
    conn = ibm_db.connect(conn_str, "", "")   

    # Query source products table
    query = """
    SELECT product_id, name, description, brand, category, subcategory,
           price, color, size, in_stock, rating, reviews
    FROM source_products
    ORDER BY product_id
    """

    stmt = ibm_db.exec_immediate(conn, query)   

    documents = []
    row = ibm_db.fetch_tuple(stmt)   

    while row:
        product_id, name, description, brand, category, subcategory, price, color, size, in_stock, rating, reviews = row

        # STAGE 1: Create Haystack Document WITH metadata for structured filtering
        # Metadata enables exact constraint filtering (price, brand, color, etc.)
        # Content (description) enables semantic search via embeddings
        # This hybrid approach combines structured + semantic search
        doc = Document(
            id=product_id,
            content=description,  # Rich description for semantic search
            meta={
                # Structured fields for exact filtering
                "name": name,
                "brand": brand,
                "category": category,
                "subcategory": subcategory,
                "price": float(price),  # Numeric for range queries
                "color": color,
                "size": size,
                "in_stock": bool(in_stock),  # Boolean for availability
                "rating": float(rating),  # Numeric for quality filtering
                "reviews": int(reviews),  # Numeric for popularity
            },
        )
        documents.append(doc)
        row = ibm_db.fetch_tuple(stmt)   

    ibm_db.close(conn)   

    print(colored(f"  ✓ Loaded {len(documents)} products from source table", Colors.GREEN))
    print(f"\n  {bold('Sample Documents:')}")
    for i, doc in enumerate(documents[:3], 1):
        print(f"    {i}. {doc.content[:80]}...")
    if len(documents) > 3:
        print(f"    ... and {len(documents) - 3} more")

    return documents


def setup_vector_store() -> DB2DocumentStore:
    """
    Step 3: Initialize DB2 vector store (separate from source table).

    This creates a specialized table optimized for vector search.
    
    PRODUCTION NOTE: For production use, set recreate_table=False to preserve existing data.
    See commented code below for incremental sync strategies.
    """
    print_subsection("Step 3: Setting up Vector Store")

    print(f"  • Database: {bold('TESTDB')}")
    print(f"  • Vector table: {bold('product_vectors')}")
    print(f"  • Embedding dimension: {bold('384')} (all-MiniLM-L6-v2)")
    print(f"  • Distance metric: {bold('cosine')}")

    # ============================================================================
    # DEVELOPMENT MODE: Drops and recreates table (LOSES ALL DATA)
    # ============================================================================
    document_store = DB2DocumentStore(
        database="TESTDB",
        username=Secret.from_env_var("DB2_USER", strict=False) or Secret.from_token("db2inst1"),
        password=Secret.from_env_var("DB2_PASSWORD", strict=False) or Secret.from_token("password"),
        table_name="product_vectors",
        embedding_dimension=384,
        distance_metric="cosine",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Track embedding model
        recreate_table=True,  #  DEVELOPMENT ONLY - drops entire table!
    )

    # ============================================================================
    # PRODUCTION MODE: Preserves existing data (RECOMMENDED)
    # ============================================================================
    # Uncomment this block for production use:
    #
    # document_store = DB2DocumentStore(
    #     database="TESTDB",
    #     username=Secret.from_env_var("DB2_USER", strict=False) or Secret.from_token("db2inst1"),
    #     password=Secret.from_env_var("DB2_PASSWORD", strict=False) or Secret.from_token("password"),
    #     table_name="product_vectors",
    #     embedding_dimension=384,
    #     distance_metric="cosine",
    #     embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    #     recreate_table=False,  #  PRODUCTION - keeps existing data
    # )

    print(colored("  ✓ Vector store initialized successfully", Colors.GREEN))
    return document_store


def index_products_with_embeddings(document_store: DB2DocumentStore, documents: list[Document]) -> dict:
    """
    Step 4: Generate embeddings and index into vector store.

    This creates vector representations of your products for semantic search.
    
    PRODUCTION NOTE: For incremental updates, see commented code below for
    strategies to sync only new/changed documents.
    """
    print_subsection("Step 4: Generating Embeddings and Indexing")

    print(f"  • Embedding model: {bold('all-MiniLM-L6-v2')}")
    print(f"  • Documents to process: {bold(str(len(documents)))}")

    # ============================================================================
    # BASIC MODE: Index all documents (suitable for initial load or full refresh)
    # ============================================================================
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

    # ============================================================================
    # PRODUCTION MODE: Incremental sync (only new/changed documents)
    # ============================================================================
    # Uncomment this block for production incremental updates:
    #
    # from haystack.document_stores.types import DuplicatePolicy
    #
    # # Strategy 1: Detect new documents only
    # print("  • Detecting new documents...")
    # existing_docs = document_store.filter_documents(filters={})
    # existing_ids = {doc.id for doc in existing_docs}
    # new_docs = [doc for doc in documents if doc.id not in existing_ids]
    #
    # if new_docs:
    #     print(f"  • Found {len(new_docs)} new products to index")
    #     result = indexing_pipeline.run({"embedder": {"documents": new_docs}})
    #     print(colored(f"  ✓ Added {len(new_docs)} new products", Colors.GREEN))
    # else:
    #     print(colored("  ✓ No new products to add", Colors.YELLOW))
    #     result = {"writer": {"documents_written": 0}}
    #
    # # Strategy 2: Upsert all (updates existing, adds new)
    # # This is simpler but processes all documents every time
    # # result = indexing_pipeline.run({"embedder": {"documents": documents}})
    # # document_store.write_documents(
    # #     result["embedder"]["documents"],
    # #     policy=DuplicatePolicy.OVERWRITE  # Updates existing docs
    # # )
    #
    # # Strategy 3: Timestamp-based sync (requires timestamp column in source table)
    # # Query source table: SELECT * FROM source_products WHERE updated_at > ?
    # # Only process documents that changed since last sync
    # # Store last_sync_time in metadata table or config file

    return result


def create_hybrid_search_pipeline(document_store: DB2DocumentStore) -> Pipeline:
    """
    Step 5: Create hybrid search pipeline.

    Combines vector similarity search with keyword search for best results.
    """
    print_subsection("Step 5: Creating Hybrid Search Pipeline")

    print(f"  • Text Embedder: {bold('SentenceTransformers')} (all-MiniLM-L6-v2)")
    print(f"  • Embedding Retriever: {bold('DB2EmbeddingRetriever')} (top_k=5)")
    print(f"  • Keyword Retriever: {bold('DB2KeywordRetriever')} (top_k=5)")
    print(f"  • Document Joiner: {bold('Reciprocal Rank Fusion')} (top_k=10)")

    pipeline = Pipeline()

    # Add components
    pipeline.add_component(
        "text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    )
    pipeline.add_component("embedding_retriever", DB2EmbeddingRetriever(document_store=document_store, top_k=5))
    pipeline.add_component("keyword_retriever", DB2KeywordRetriever(document_store=document_store, top_k=5))
    pipeline.add_component("joiner", DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=10))

    # Connect components
    pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    pipeline.connect("embedding_retriever.documents", "joiner.documents")
    pipeline.connect("keyword_retriever.documents", "joiner.documents")

    print(colored("  ✓ Pipeline created and connected successfully", Colors.GREEN))
    return pipeline


def parse_query_with_filters(user_query: str) -> tuple[str, dict | None]:
    """
    STAGE 3: Intelligent query parser with regex/rules.
    
    Extracts structured filters from natural language queries using lightweight
    regex patterns. Falls back to pure semantic search when patterns don't match.
    
    Strategy: Parse only obvious patterns, fallback to semantic search when uncertain.
    This creates a robust hybrid system that doesn't break on unexpected input.
    
    Examples:
        "black Nike shoes under $150" → 
            query="shoes", filters={"brand": "Nike", "color": "black", "price": {"$lt": 150}}
        
        "comfortable running shoes" → 
            query="comfortable running shoes", filters=None (pure semantic)
    
    Args:
        user_query: Natural language query from user
        
    Returns:
        Tuple of (semantic_query, filters_dict)
    """
    filters = {}
    remaining_query = user_query.lower()
    
    # Pattern 1: Extract brand names (Nike, Adidas)
    brand_pattern = r'\b(nike|adidas)\b'
    brand_match = re.search(brand_pattern, remaining_query, re.IGNORECASE)
    if brand_match:
        brand = brand_match.group(1).capitalize()
        filters["brand"] = brand
        remaining_query = re.sub(brand_pattern, '', remaining_query, flags=re.IGNORECASE)
    
    # Pattern 2: Extract colors (black, white, blue, pink, gray)
    color_pattern = r'\b(black|white|blue|pink|gray|grey|red|green)\b'
    color_match = re.search(color_pattern, remaining_query, re.IGNORECASE)
    if color_match:
        color = color_match.group(1).lower()
        if color == "grey":
            color = "gray"  # Normalize spelling
        filters["color"] = color
        remaining_query = re.sub(color_pattern, '', remaining_query, flags=re.IGNORECASE)
    
    # Pattern 3: Extract price constraints
    # "under $150", "below $200", "less than $100"
    price_under_pattern = r'(?:under|below|less than|cheaper than)\s*\$?(\d+)'
    price_under_match = re.search(price_under_pattern, remaining_query, re.IGNORECASE)
    if price_under_match:
        price_limit = int(price_under_match.group(1))
        filters["price"] = {"$lt": price_limit}
        remaining_query = re.sub(price_under_pattern, '', remaining_query, flags=re.IGNORECASE)
    
    # "over $100", "above $150", "more than $200"
    price_over_pattern = r'(?:over|above|more than|expensive than)\s*\$?(\d+)'
    price_over_match = re.search(price_over_pattern, remaining_query, re.IGNORECASE)
    if price_over_match:
        price_limit = int(price_over_match.group(1))
        if "price" in filters:
            # Combine with existing price filter (range query)
            filters["price"]["$gte"] = price_limit
        else:
            filters["price"] = {"$gte": price_limit}
        remaining_query = re.sub(price_over_pattern, '', remaining_query, flags=re.IGNORECASE)
    
    # "between $100 and $150", "$100-$150", "$100 to $150"
    price_range_pattern = r'(?:between\s*)?\$?(\d+)(?:\s*(?:and|to|-)\s*)\$?(\d+)'
    price_range_match = re.search(price_range_pattern, remaining_query, re.IGNORECASE)
    if price_range_match and not price_under_match and not price_over_match:
        price_min = int(price_range_match.group(1))
        price_max = int(price_range_match.group(2))
        filters["price"] = {"$gte": price_min, "$lte": price_max}
        remaining_query = re.sub(price_range_pattern, '', remaining_query, flags=re.IGNORECASE)
    
    # Pattern 4: Extract availability
    # "in stock", "available", "currently available"
    stock_pattern = r'\b(?:in stock|available|currently available)\b'
    if re.search(stock_pattern, remaining_query, re.IGNORECASE):
        filters["in_stock"] = True
        remaining_query = re.sub(stock_pattern, '', remaining_query, flags=re.IGNORECASE)
    
    # Pattern 5: Extract rating
    # "highly rated", "top rated", "4+ stars", "rating above 4.5"
    rating_pattern = r'(?:rating\s*(?:above|over|>=?)\s*)?(\d+(?:\.\d+)?)\+?\s*(?:stars?)?'
    rating_match = re.search(rating_pattern, remaining_query, re.IGNORECASE)
    if rating_match and ("rating" in remaining_query or "rated" in remaining_query or "stars" in remaining_query):
        rating_value = float(rating_match.group(1))
        filters["rating"] = {"$gte": rating_value}
        remaining_query = re.sub(rating_pattern, '', remaining_query, flags=re.IGNORECASE)
    
    # Clean up remaining query (remove extra spaces, punctuation)
    remaining_query = re.sub(r'\s+', ' ', remaining_query).strip()
    remaining_query = re.sub(r'[,\.]', '', remaining_query).strip()
    
    # If no semantic content remains, use generic term
    if not remaining_query or len(remaining_query) < 3:
        remaining_query = "shoes"  # Default semantic query
    
    # Return None for filters if empty (pure semantic search)
    final_filters = filters if filters else None
    
    return remaining_query, final_filters


def search_products(pipeline: Pipeline, query: str, filters: dict | None = None) -> list[Document]:
    """
    Step 6: Perform hybrid search with optional filters.
    """
    print(f"\n{colored('─' * 70, Colors.CYAN)}")
    print(f"  {bold('Query:')} {colored(query, Colors.YELLOW)}")
    if filters:
        print(f"  {bold('Filters:')} {filters}")
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

    # Display results
    print(f"\n  {colored(f'Found {len(documents)} products:', Colors.GREEN + Colors.BOLD)}\n")

    for i, doc in enumerate(documents, 1):
        # Display full content (which now contains all product info)
        print(f"  {colored(f'{i}.', Colors.BOLD)} {doc.content[:100]}...")
        
        score_color = Colors.GREEN if doc.score > 0.8 else Colors.YELLOW if doc.score > 0.5 else Colors.RED
        print(f"     Relevance Score: {colored(f'{doc.score:.4f}', score_color)}")
        print()

    return documents


def main() -> None:
    """Main function demonstrating the complete workflow."""
    print_section("DB2 Product Search: From Source Table to Vector Search")
    print(f"\n{colored('Complete workflow: Source Table → Load → Embed → Search', Colors.CYAN)}")

    # Database credentials
    database = "TESTDB"
    username = Secret.from_env_var("DB2_USER", strict=False) or Secret.from_token("db2inst1")
    password = Secret.from_env_var("DB2_PASSWORD", strict=False) or Secret.from_token("password")

    username_str = username.resolve_value() or "db2inst1"
    password_str = password.resolve_value() or "password"

    # Step 1: Create source products table (simulating existing business data)
    create_source_products_table(database, username_str, password_str)

    # Step 2: Load products from source table
    documents = load_products_from_db2(database, username_str, password_str)

    # Step 3: Setup vector store (separate table for embeddings)
    document_store = setup_vector_store()

    # Step 4: Generate embeddings and index
    index_products_with_embeddings(document_store, documents)

    # Step 5: Create search pipeline
    pipeline = create_hybrid_search_pipeline(document_store)

    # Step 6: Perform searches
    print_section("SCENARIO 1: Simple Semantic Search")
    print(colored("  Natural language query without filters", Colors.CYAN))
    # search_products(pipeline, "light sneakers for everyday use")
    search_products(pipeline, "running shoes")

    print_section("SCENARIO 2: Brand-Specific Search")
    print(colored("  Natural language query mentioning brand", Colors.CYAN))
    search_products(pipeline, "Nike running shoes")

    print_section("SCENARIO 3: Price-Conscious Search")
    print(colored("  Query for affordable options", Colors.CYAN))
    search_products(pipeline, "affordable running shoes under $150")

    print_section("SCENARIO 4: Feature-Based Search")
    print(colored("  Finding products by describing features", Colors.CYAN))
    search_products(pipeline, "lightweight cushioning for speed and performance")

    print_section("SCENARIO 5: Availability Search")
    print(colored("  Query for in-stock products", Colors.CYAN))
    search_products(pipeline, "running shoes currently in stock")

    # STAGE 2: Filter-based search scenarios
    print_section("SCENARIO 6: Price Range Filter")
    print(colored("  Structured filter: Products between $100-$150", Colors.CYAN))
    filters = {"price": {"$gte": 100, "$lte": 150}}
    search_products(pipeline, "running shoes", filters=filters)

    print_section("SCENARIO 7: Brand Filter")
    print(colored("  Structured filter: Nike products only", Colors.CYAN))
    filters = {"brand": "Nike"}
    search_products(pipeline, "running shoes", filters=filters)

    print_section("SCENARIO 8: Color Filter")
    print(colored("  Structured filter: Black shoes", Colors.CYAN))
    filters = {"color": "black"}
    search_products(pipeline, "shoes", filters=filters)

    print_section("SCENARIO 9: Complex Multi-Filter")
    print(colored("  Combined filters: Nike, under $150, in stock", Colors.CYAN))
    filters = {
        "$and": [
            {"brand": "Nike"},
            {"price": {"$lt": 150}},
            {"in_stock": True}
        ]
    }
    search_products(pipeline, "running shoes", filters=filters)

    print_section("SCENARIO 10: High-Rated Products")
    print(colored("  Structured filter: Rating >= 4.5 stars", Colors.CYAN))
    filters = {"rating": {"$gte": 4.5}}
    search_products(pipeline, "shoes", filters=filters)

    # STAGE 3: Intelligent query parsing scenarios
    print_section("SCENARIO 11: Intelligent Parse - Brand + Color + Price")
    print(colored("  Natural language: 'black Nike shoes under $150'", Colors.CYAN))
    user_query = "black Nike shoes under $150"
    parsed_query, parsed_filters = parse_query_with_filters(user_query)
    print(f"  {colored('→ Parsed Query:', Colors.YELLOW)} {parsed_query}")
    print(f"  {colored('→ Extracted Filters:', Colors.YELLOW)} {parsed_filters}")
    search_products(pipeline, parsed_query, filters=parsed_filters)

    print_section("SCENARIO 12: Intelligent Parse - Price Range")
    print(colored("  Natural language: 'running shoes between $100 and $150'", Colors.CYAN))
    user_query = "running shoes between $100 and $150"
    parsed_query, parsed_filters = parse_query_with_filters(user_query)
    print(f"  {colored('→ Parsed Query:', Colors.YELLOW)} {parsed_query}")
    print(f"  {colored('→ Extracted Filters:', Colors.YELLOW)} {parsed_filters}")
    search_products(pipeline, parsed_query, filters=parsed_filters)

    print_section("SCENARIO 13: Intelligent Parse - Availability")
    print(colored("  Natural language: 'white Adidas shoes in stock'", Colors.CYAN))
    user_query = "white Adidas shoes in stock"
    parsed_query, parsed_filters = parse_query_with_filters(user_query)
    print(f"  {colored('→ Parsed Query:', Colors.YELLOW)} {parsed_query}")
    print(f"  {colored('→ Extracted Filters:', Colors.YELLOW)} {parsed_filters}")
    search_products(pipeline, parsed_query, filters=parsed_filters)

    print_section("SCENARIO 14: Fallback to Semantic Search")
    print(colored("  Natural language: 'comfortable cushioning for daily wear'", Colors.CYAN))
    user_query = "comfortable cushioning for daily wear"
    parsed_query, parsed_filters = parse_query_with_filters(user_query)
    print(f"  {colored('→ Parsed Query:', Colors.YELLOW)} {parsed_query}")
    print(f"  {colored('→ Extracted Filters:', Colors.YELLOW)} {parsed_filters or 'None (pure semantic search)'}")
    search_products(pipeline, parsed_query, filters=parsed_filters)

    # Summary
    print_section("Workflow Complete!")
    print(f"\n{bold('Workflow Steps Demonstrated:')}")
    print(f"  1. {colored('Source Table Creation', Colors.GREEN)} - Traditional relational product data")
    print(f"  2. {colored('Data Loading', Colors.GREEN)} - Reading from existing DB2 table")
    print(f"  3. {colored('Vector Store Setup', Colors.GREEN)} - Specialized table for embeddings")
    print(f"  4. {colored('Embedding Generation', Colors.GREEN)} - Converting text to vectors")
    print(f"  5. {colored('Hybrid Search', Colors.GREEN)} - Vector + keyword retrieval")

    print(f"\n{bold('Production Benefits:')}")
    print("  • Source data remains unchanged in original tables")
    print("  • Vector store can be rebuilt/updated independently")
    print("  • Hybrid approach: Semantic search + Structured filtering")
    print("  • Metadata enables exact constraint queries (price, brand, etc.)")
    print("  • Rich descriptions enable semantic relevance")

    print(f"\n{bold('Next Steps for Production:')}")
    print("  1. Set up incremental updates (sync source → vector store)")
    print("  2. Add batch processing for large catalogs")
    print("  3. Implement caching for frequently searched queries")
    print("  4. Monitor and optimize embedding generation")
    print("  5. Add A/B testing for search relevance")

    print(f"\n{colored('=' * 70, Colors.CYAN)}\n")


if __name__ == "__main__":
    main()

# Made with Bob
