# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
DB2 Pure Vector Search + Hybrid Search (Vector + SQL Filters)

This example demonstrates:

1. Pure semantic vector search
2. Hybrid search (vector similarity + SQL metadata filters)
3. Intelligent query parsing
4. Cosine similarity scoring
5. Score threshold filtering

Key Design Goals:
- Some products are semantically CLOSE
- Some products are semantically FAR
- Metadata appears naturally inside descriptions
- Semantic search and SQL filtering work together

Requirements:
`pip install sentence-transformers haystack-ai ibm-db python-dotenv`
"""

import os
import logging
import re
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv
from haystack import Document, Pipeline
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret

from haystack_integrations.components.retrievers.db2 import (
    DB2EmbeddingRetriever,
)
from haystack_integrations.document_stores.db2 import (
    DB2DocumentStore,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Optional:
# Set HF_TOKEN in environment to remove unauthenticated warnings
# os.environ["HF_TOKEN"] = "your_token"

warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    message=".*tokenizer_kwargs.*"
)
# Suppress Haystack logs
logging.getLogger("haystack").setLevel(logging.ERROR)

# Suppress SentenceTransformer logs
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Suppress Transformers logs
logging.getLogger("transformers").setLevel(logging.ERROR)

# Suppress HuggingFace Hub logs
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Suppress HTTP request logs
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# Suppress asyncio noise
logging.getLogger("asyncio").setLevel(logging.ERROR)


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# -----------------------------------------------------------------------------
# Load .env
# -----------------------------------------------------------------------------

try:
    env_path = Path(__file__).parent.parent / ".env"

    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

except ImportError:
    pass

# -----------------------------------------------------------------------------
# Terminal Colors
# -----------------------------------------------------------------------------


class Colors:
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"

    @staticmethod
    def supported():
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def colored(text, color):
    if Colors.supported():
        return f"{color}{text}{Colors.END}"
    return text


def print_section(title):
    print("\n" + "=" * 80)
    print(colored(title, Colors.BOLD + Colors.CYAN))
    print("=" * 80)


# -----------------------------------------------------------------------------
# Create Source Table
# -----------------------------------------------------------------------------


def create_source_products_table(database, username, password):

    import ibm_db

    use_ssl = os.getenv("DB2_SSL_ENABLED", "").lower() in {"1", "true", "yes"}
    hostname = os.getenv("DB2_HOSTNAME")
    port = os.getenv("DB2_SSL_PORT", "50001") if use_ssl else os.getenv("DB2_PORT", "50000")

    conn_str = (
        f"DATABASE={database};"
        f"HOSTNAME={hostname};"
        f"PORT={port};"
        f"PROTOCOL=TCPIP;"
        f"UID={username};"
        f"PWD={password};"
        f"{'SECURITY=SSL;' if use_ssl else ''}"
    )

    conn = ibm_db.connect(conn_str, "", "")

    try:
        ibm_db.exec_immediate(conn, "DROP TABLE source_products")
    except Exception:
        pass

    create_table_sql = """
    CREATE TABLE source_products (
        product_id VARCHAR(50) NOT NULL PRIMARY KEY,
        name VARCHAR(200),
        description VARCHAR(3000),
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

    # -------------------------------------------------------------------------
    # Product Dataset
    #
    # IMPORTANT:
    # - Some products intentionally close semantically
    # - Some intentionally far apart
    # -------------------------------------------------------------------------

    products = [

        # ---------------------------------------------------------------------
        # RUNNING / ENDURANCE CLUSTER
        # ---------------------------------------------------------------------

        (
            "nike_run_001",
            "Nike Pegasus Marathon",
            """
            Nike Pegasus Marathon is a lightweight running shoe priced at $140.
            Designed specifically for long-distance road running, marathon training,
            and endurance workouts. Features responsive foam cushioning and breathable
            mesh upper for runners training daily. Excellent for speed workouts,
            cardio sessions, and half-marathon preparation. Black Nike running shoes
            available in size 10 and currently in stock. Rated 4.7 stars by 1800 runners.
            """,
            "Nike",
            "shoes",
            "running",
            140.00,
            "black",
            "10",
            True,
            4.7,
            1800,
        ),

        (
            "adidas_run_001",
            "Adidas Boston Speed",
            """
            Adidas Boston Speed is a blue lightweight running shoe priced at $150.
            Built for tempo runs, endurance training, and race preparation.
            Features responsive cushioning and breathable performance upper for
            long-distance athletes. Ideal for marathon runners seeking lightweight
            speed training footwear. Adidas running shoes currently in stock.
            Rated 4.6 stars by 1500 athletes.
            """,
            "Adidas",
            "shoes",
            "running",
            150.00,
            "blue",
            "10",
            True,
            4.6,
            1500,
        ),

        (
            "nike_run_002",
            "Nike Recovery Comfort",
            """
            Nike Recovery Comfort is a gray cushioned running shoe priced at $160.
            Designed for recovery runs, daily jogging, and injury prevention.
            Soft foam cushioning reduces stress on knees and ankles during long
            training sessions. Great for runners prioritizing comfort over speed.
            Nike running footwear available in stock with 4.5-star rating.
            """,
            "Nike",
            "shoes",
            "running",
            160.00,
            "gray",
            "10",
            True,
            4.5,
            1200,
        ),

        # ---------------------------------------------------------------------
        # BASKETBALL CLUSTER
        # ---------------------------------------------------------------------

        (
            "adidas_basket_001",
            "Adidas Court Pro",
            """
            Adidas Court Pro is a white basketball sneaker priced at $170.
            High-top athletic footwear engineered for indoor basketball courts,
            jump shots, lateral movement, and explosive takeoffs. Excellent ankle
            support and court grip for competitive players. Designed for basketball
            tournaments, practice sessions, and indoor sports performance.
            Rated 4.8 stars by 900 basketball athletes.
            """,
            "Adidas",
            "shoes",
            "basketball",
            170.00,
            "white",
            "11",
            True,
            4.8,
            900,
        ),

        (
            "nike_basket_001",
            "Nike Dunk Elite",
            """
            Nike Dunk Elite is a red basketball shoe priced at $190.
            Built for fast court movement, slam dunks, and professional indoor
            basketball games. Responsive cushioning absorbs impact during jumping.
            Durable outsole improves grip on hardwood courts. Premium Nike
            basketball sneakers rated 4.7 stars and currently available.
            """,
            "Nike",
            "shoes",
            "basketball",
            190.00,
            "red",
            "10",
            True,
            4.7,
            850,
        ),

        # ---------------------------------------------------------------------
        # HIKING / OUTDOOR CLUSTER
        # ---------------------------------------------------------------------

        (
            "merrell_hike_001",
            "Merrell Mountain Trek",
            """
            Merrell Mountain Trek is a brown waterproof hiking boot priced at $210.
            Designed for mountain trails, trekking, camping, and rugged outdoor
            adventures. Waterproof Gore-Tex lining protects feet during rain and
            river crossings. Aggressive outsole improves traction on rocky terrain.
            Ideal for backpacking and wilderness exploration. Rated 4.9 stars by hikers.
            """,
            "Merrell",
            "shoes",
            "hiking",
            210.00,
            "brown",
            "11",
            True,
            4.9,
            2200,
        ),

        (
            "columbia_hike_001",
            "Columbia Trail Explorer",
            """
            Columbia Trail Explorer is a gray outdoor hiking shoe priced at $180.
            Lightweight trekking footwear for forest trails, camping trips, and
            nature walks. Shock-absorbing sole and waterproof protection make it
            ideal for outdoor terrain and adventure travel. Columbia hiking shoes
            currently in stock with 4.6-star customer rating.
            """,
            "Columbia",
            "shoes",
            "hiking",
            180.00,
            "gray",
            "10",
            True,
            4.6,
            1400,
        ),

        # ---------------------------------------------------------------------
        # FORMAL / BUSINESS CLUSTER
        # ---------------------------------------------------------------------

        (
            "clarks_formal_001",
            "Clarks Executive Oxford",
            """
            Clarks Executive Oxford is a black leather formal shoe priced at $145.
            Professional business footwear designed for office meetings, weddings,
            formal events, and corporate environments. Genuine leather upper with
            polished finish suitable for suits and executive attire. Comfortable
            insole supports all-day office wear. Rated 4.7 stars by professionals.
            """,
            "Clarks",
            "shoes",
            "formal",
            145.00,
            "black",
            "10",
            True,
            4.7,
            980,
        ),

        # ---------------------------------------------------------------------
        # WINTER / SNOW CLUSTER
        # ---------------------------------------------------------------------

        (
            "sorel_winter_001",
            "Sorel Arctic Shield",
            """
            Sorel Arctic Shield is a black insulated winter boot priced at $220.
            Heavy-duty cold-weather footwear for snowstorms, icy roads, and freezing
            winter temperatures. Thermal insulation keeps feet warm below -30 degrees.
            Waterproof shell protects during snow and slush conditions. Ideal for
            winter hiking and snow travel. Currently out of stock.
            """,
            "Sorel",
            "shoes",
            "winter",
            220.00,
            "black",
            "11",
            False,
            4.5,
            600,
        ),
    ]

    insert_sql = """
    INSERT INTO source_products
    (
        product_id,
        name,
        description,
        brand,
        category,
        subcategory,
        price,
        color,
        size,
        in_stock,
        rating,
        reviews
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    stmt = ibm_db.prepare(conn, insert_sql)

    for product in products:
        ibm_db.execute(stmt, product)

    print(colored("Inserted products into source_products", Colors.GREEN))

    ibm_db.close(conn)


# -----------------------------------------------------------------------------
# Load Products
# -----------------------------------------------------------------------------


def load_products(database, username, password):
    """Load products from source DB2 table."""
    print_section("INGESTION STAGE 1: Loading Existing Table")
    print(colored("  ├─ Connecting to source DB2 table...", Colors.CYAN))

    import ibm_db

    use_ssl = os.getenv("DB2_SSL_ENABLED", "").lower() in {"1", "true", "yes"}
    hostname = os.getenv("DB2_HOSTNAME")
    port = os.getenv("DB2_SSL_PORT", "50001") if use_ssl else os.getenv("DB2_PORT", "50000")

    conn_str = (
        f"DATABASE={database};"
        f"HOSTNAME={hostname};"
        f"PORT={port};"
        f"PROTOCOL=TCPIP;"
        f"UID={username};"
        f"PWD={password};"
        f"{'SECURITY=SSL;' if use_ssl else ''}"
    )

    conn = ibm_db.connect(conn_str, "", "")

    query = """
    SELECT
        product_id,
        name,
        description,
        brand,
        category,
        subcategory,
        price,
        color,
        size,
        in_stock,
        rating,
        reviews
    FROM source_products
    """

    stmt = ibm_db.exec_immediate(conn, query)

    documents = []

    row = ibm_db.fetch_tuple(stmt)

    while row:

        (
            product_id,
            name,
            description,
            brand,
            category,
            subcategory,
            price,
            color,
            size,
            in_stock,
            rating,
            reviews,
        ) = row

        doc = Document(
            id=product_id,
            content=description,
            meta={
                "name": name,
                "brand": brand,
                "category": category,
                "subcategory": subcategory,
                "price": float(price),
                "color": color,
                "size": size,
                "in_stock": bool(in_stock),
                "rating": float(rating),
                "reviews": int(reviews),
            },
        )

        documents.append(doc)

        row = ibm_db.fetch_tuple(stmt)

    ibm_db.close(conn)

    return documents


# -----------------------------------------------------------------------------
# Vector Store
# -----------------------------------------------------------------------------


def setup_vector_store():

    use_ssl = os.getenv("DB2_SSL_ENABLED", "").lower() in {"1", "true", "yes"}
    port = int(os.getenv("DB2_SSL_PORT", "50001")) if use_ssl else int(os.getenv("DB2_PORT", "50000"))

    document_store = DB2DocumentStore(
        database=os.getenv("DB2_DATABASE", "TESTDB"),
        hostname=os.getenv("DB2_HOSTNAME"),
        port=port,
        username=Secret.from_env_var("DB2_USER"),
        password=Secret.from_env_var("DB2_PASSWORD"),
        table_name="product_vectors",
        embedding_dimension=384,
        distance_metric="cosine",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        recreate_table=True,
    )

    return document_store


# -----------------------------------------------------------------------------
# Index Documents
# -----------------------------------------------------------------------------


def index_documents(document_store, documents):
    """Index documents with embeddings into vector store."""
    print_section("INGESTION STAGE 2: Embedding & Vector Generation")
    print(colored("  ├─ Embedding Model: sentence-transformers/all-MiniLM-L6-v2", Colors.CYAN))
    print(colored(f"  ├─ Processing {len(documents)} documents...", Colors.CYAN))

    indexing_pipeline = Pipeline()

    indexing_pipeline.add_component(
        "embedder",
        SentenceTransformersDocumentEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        ),
    )

    indexing_pipeline.add_component(
        "writer",
        DocumentWriter(document_store=document_store),
    )

    indexing_pipeline.connect("embedder", "writer")

    indexing_pipeline.run(
        {
            "embedder": {
                "documents": documents
            }
        }
    )

    print(colored("Documents indexed successfully", Colors.GREEN))


# -----------------------------------------------------------------------------
# Search Pipeline
# -----------------------------------------------------------------------------


def create_search_pipeline(document_store):

    pipeline = Pipeline()

    pipeline.add_component(
        "text_embedder",
        SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        ),
    )

    pipeline.add_component(
        "embedding_retriever",
        DB2EmbeddingRetriever(
            document_store=document_store,
            top_k=3,
        ),
    )

    pipeline.connect(
        "text_embedder.embedding",
        "embedding_retriever.query_embedding",
    )

    return pipeline


# -----------------------------------------------------------------------------
# Intelligent Query Parser
# -----------------------------------------------------------------------------

def parse_query_with_filters(user_query):
    """
    Intelligent hybrid query parser.

    Extracts:
    - brand
    - color
    - price constraints
    - stock availability
    - ratings

    Remaining text becomes semantic vector query.
    """

    filters = {}

    # -------------------------------------------------------------------------
    # Normalize Query
    # -------------------------------------------------------------------------

    remaining_query = user_query.lower().strip()

    # Remove punctuation
    remaining_query = re.sub(r"[,\.\?!]", " ", remaining_query)

    # Normalize spaces
    remaining_query = re.sub(r"\s+", " ", remaining_query)

    # -------------------------------------------------------------------------
    # Synonym Normalization
    # -------------------------------------------------------------------------

    SYNONYM_MAP = {
        "sneakers": "shoes",
        "trainer": "running shoes",
        "trainers": "running shoes",
        "footwear": "shoes",
        "kicks": "shoes",
        "boots": "boots",
    }

    for source, target in SYNONYM_MAP.items():
        remaining_query = re.sub(
            rf"\b{source}\b",
            target,
            remaining_query,
            flags=re.IGNORECASE,
        )

    # -------------------------------------------------------------------------
    # Supported Brands
    # -------------------------------------------------------------------------

    SUPPORTED_BRANDS = [
        "nike",
        "adidas",
        "merrell",
        "columbia",
        "clarks",
        "sorel",
    ]

    brand_pattern = (
        r"\b(" + "|".join(SUPPORTED_BRANDS) + r")\b"
    )

    brand_match = re.search(
        brand_pattern,
        remaining_query,
        re.IGNORECASE,
    )

    if brand_match:

        brand = brand_match.group(1)
        brand = brand[0].upper() + brand[1:].lower()

        filters["brand"] = brand

        remaining_query = re.sub(
            brand_pattern,
            "",
            remaining_query,
            flags=re.IGNORECASE,
        )

    # -------------------------------------------------------------------------
    # Color Extraction
    # -------------------------------------------------------------------------

    SUPPORTED_COLORS = [
        "black",
        "white",
        "blue",
        "gray",
        "grey",
        "red",
        "green",
        "brown",
        "pink",
    ]

    color_pattern = (
        r"\b(" + "|".join(SUPPORTED_COLORS) + r")\b"
    )

    color_match = re.search(
        color_pattern,
        remaining_query,
        re.IGNORECASE,
    )

    if color_match:

        color = color_match.group(1).lower()

        # Normalize spelling
        if color == "grey":
            color = "gray"

        filters["color"] = color

        remaining_query = re.sub(
            color_pattern,
            "",
            remaining_query,
            flags=re.IGNORECASE,
        )

    # -------------------------------------------------------------------------
    # PRICE RANGE
    #
    # Examples:
    # - between $100 and $200
    # - $100 to $200
    # - $100-$200
    # -------------------------------------------------------------------------

    price_range_pattern = (
        r"(?:between\s*)?"
        r"\$?(\d+)"
        r"(?:\s*(?:and|to|-)\s*)"
        r"\$?(\d+)"
    )

    price_range_match = re.search(
        price_range_pattern,
        remaining_query,
        re.IGNORECASE,
    )

    if price_range_match:

        min_price = int(price_range_match.group(1))
        max_price = int(price_range_match.group(2))

        filters["price"] = {
            "$gte": min_price,
            "$lte": max_price,
        }

        remaining_query = re.sub(
            price_range_pattern,
            "",
            remaining_query,
            flags=re.IGNORECASE,
        )

    else:

        # ---------------------------------------------------------------------
        # PRICE UNDER
        #
        # Examples:
        # - under $150
        # - below 200
        # - less than 100
        # ---------------------------------------------------------------------

        price_under_pattern = (
            r"(?:under|below|less than|cheaper than)"
            r"\s*\$?(\d+)"
        )

        price_under_match = re.search(
            price_under_pattern,
            remaining_query,
            re.IGNORECASE,
        )

        if price_under_match:

            price_limit = int(price_under_match.group(1))

            filters["price"] = {
                "$lt": price_limit
            }

            remaining_query = re.sub(
                price_under_pattern,
                "",
                remaining_query,
                flags=re.IGNORECASE,
            )

        # ---------------------------------------------------------------------
        # PRICE OVER
        #
        # Examples:
        # - over $100
        # - above $200
        # - more than $300
        # ---------------------------------------------------------------------

        price_over_pattern = (
            r"(?:over|above|more than|greater than)"
            r"\s*\$?(\d+)"
        )

        price_over_match = re.search(
            price_over_pattern,
            remaining_query,
            re.IGNORECASE,
        )

        if price_over_match:

            price_limit = int(price_over_match.group(1))

            if "price" in filters:
                filters["price"]["$gte"] = price_limit
            else:
                filters["price"] = {
                    "$gte": price_limit
                }

            remaining_query = re.sub(
                price_over_pattern,
                "",
                remaining_query,
                flags=re.IGNORECASE,
            )

    # -------------------------------------------------------------------------
    # STOCK AVAILABILITY
    #
    # Examples:
    # - in stock
    # - available
    # - currently available
    # -------------------------------------------------------------------------

    stock_pattern = (
        r"\b(?:in stock|available|currently available)\b"
    )

    if re.search(
        stock_pattern,
        remaining_query,
        re.IGNORECASE,
    ):

        filters["in_stock"] = True

        remaining_query = re.sub(
            stock_pattern,
            "",
            remaining_query,
            flags=re.IGNORECASE,
        )

    # -------------------------------------------------------------------------
    # RATING FILTER
    #
    # Examples:
    # - rating above 4.5
    # - 4+ stars
    # - 4.5 stars
    # - rated above 4
    # -------------------------------------------------------------------------

    rating_pattern = (
        r"(?:rating\s*(?:above|over|>=?)\s*)?"
        r"(\d+(?:\.\d+)?)"
        r"\+?\s*(?:stars?)?"
    )

    rating_match = re.search(
        rating_pattern,
        remaining_query,
        re.IGNORECASE,
    )

    if (
        rating_match
        and (
            "rating" in remaining_query
            or "rated" in remaining_query
            or "stars" in remaining_query
        )
    ):

        rating_value = float(rating_match.group(1))

        filters["rating"] = {
            "$gte": rating_value
        }

        remaining_query = re.sub(
            rating_pattern,
            "",
            remaining_query,
            flags=re.IGNORECASE,
        )

    # -------------------------------------------------------------------------
    # Final Cleanup
    # -------------------------------------------------------------------------

    remaining_query = re.sub(
        r"\s+",
        " ",
        remaining_query,
    ).strip()

    # Remove leftover connector words
    remaining_query = re.sub(
        r"\b(for|with|and|show|give|provide|me|find)\b",
        "",
        remaining_query,
        flags=re.IGNORECASE,
    )

    remaining_query = re.sub(
        r"\s+",
        " ",
        remaining_query,
    ).strip()

    # -------------------------------------------------------------------------
    # Fallback Semantic Query
    # -------------------------------------------------------------------------

    if not remaining_query or len(remaining_query) < 3:
        remaining_query = "shoes"

    # -------------------------------------------------------------------------
    # Return Filters
    # -------------------------------------------------------------------------

    final_filters = filters if filters else None

    return remaining_query, final_filters

# -----------------------------------------------------------------------------
# Search
# -----------------------------------------------------------------------------


def search_products(pipeline, query, filters=None):
    print("\n" + "-" * 80)
    print(colored(f"User Query: {query}", Colors.YELLOW))

    if filters:
        print(colored(f"Filters: {filters}", Colors.CYAN))

    print("-" * 80)

    results = pipeline.run(
        {
            "text_embedder": {
                "text": query
            },
            "embedding_retriever": {
                "filters": filters
            },
        }
    )

    documents = results["embedding_retriever"]["documents"]

    # -------------------------------------------------------------------------
    # Score Threshold
    # -------------------------------------------------------------------------

    MIN_SCORE = 0.45

    documents = [
        doc
        for doc in documents
        if doc.score >= MIN_SCORE
    ]

    if not documents:
        print(colored("No matching products found", Colors.RED))
        return

    print(colored(f"\nFound {len(documents)} products\n", Colors.GREEN))

    for i, doc in enumerate(documents, 1):

        print(f"{i}. {doc.meta['name']}")
        print(f"   Brand: {doc.meta['brand']}")
        print(f"   Category: {doc.meta['subcategory']}")
        print(f"   Price: ${doc.meta['price']}")
        print(f"   Color: {doc.meta['color']}")
        print(f"   Rating: {doc.meta['rating']}")

        score_color = (
            Colors.GREEN
            if doc.score >= 0.70
            else Colors.YELLOW
        )

        print(
            f"   Similarity Score: "
            f"{colored(f'{doc.score:.4f}', score_color)}"
        )

        print(f"   Description: {doc.content[:180]}...")
        print()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():

    print_section(
        "DB2 Pure Vector Search + Hybrid Retrieval"
    )

    database = "TESTDB"

    username = Secret.from_env_var("DB2_USER")
    password = Secret.from_env_var("DB2_PASSWORD")

    username_str = username.resolve_value()
    password_str = password.resolve_value()

    # -------------------------------------------------------------------------
    # Create Source Table
    # -------------------------------------------------------------------------

    create_source_products_table(
        database,
        username_str,
        password_str,
    )

    # -------------------------------------------------------------------------
    # Load Documents
    # -------------------------------------------------------------------------

    documents = load_products(
        database,
        username_str,
        password_str,
    )

    # -------------------------------------------------------------------------
    # Setup Vector Store
    # -------------------------------------------------------------------------

    document_store = setup_vector_store()

    # -------------------------------------------------------------------------
    # Index Documents
    # -------------------------------------------------------------------------

    index_documents(document_store, documents)

    # -------------------------------------------------------------------------
    # Search Pipeline
    # -------------------------------------------------------------------------

    pipeline = create_search_pipeline(document_store)

    """Execute vector search with optional SQL filters."""
    print_section(f"RETRIEVAL PIPELINE")
    print()

    # =========================================================================
    # PURE SEMANTIC SEARCH
    # =========================================================================

    print_section("SCENARIO 1 - Pure Semantic Search")

    search_products(
        pipeline,
        "lightweight marathon running shoes",
    )

    # =========================================================================
    # HYBRID SEARCH
    # =========================================================================

    print_section("SCENARIO 2 - Hybrid Search (Vector + SQL Filters)")

    search_products(
        pipeline,
        "running shoes",
        filters={
            "price": {
                "$lt": 150
            }
        },
    )

    search_products(
        pipeline,
        "formal shoes",
        filters={
            "color": "black"
        },
    )

    # =========================================================================
    # INTELLIGENT QUERY PARSING
    # =========================================================================

    print_section("SCENARIO 3 - Intelligent Query Parsing")
    print()

    queries = [

        "black nike running shoes under $150",

        "shoes between $120 and $175",

        "give me basketball shoes for indoor courts",
    ]

    for query in queries:

        parsed_query, parsed_filters = (
            parse_query_with_filters(query)
        )

        print(colored("\nOriginal Query:", Colors.CYAN), query)
        print(colored("Parsed Semantic Query:", Colors.YELLOW), parsed_query)
        print(colored("Extracted Filters:", Colors.GREEN), parsed_filters)

        search_products(
            pipeline,
            parsed_query,
            parsed_filters,
        )

if __name__ == "__main__":
    main()