# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Bulk Insert Performance Benchmark Example

This example demonstrates the performance difference between batch insert modes
when ingesting large documents. It:

1. Loads a large text file (or generates sample data)
2. Chunks it into documents
3. Generates embeddings using a sentence transformer
4. Tests both batch insert modes (enabled/disabled)
5. Measures and compares performance
6. Tests duplicate handling scenarios

Usage:
    # With your own file:
    python bulk_insert_benchmark.py --file /path/to/large/file.txt
    
    # With generated sample data:
    python bulk_insert_benchmark.py --generate 1000
    
    # Test duplicate handling:
    python bulk_insert_benchmark.py --generate 500 --test-duplicates
"""

import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from haystack import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from sentence_transformers import SentenceTransformer

from haystack_integrations.document_stores.db2 import DB2DocumentStore

# Load environment variables
load_dotenv()


def load_and_chunk_file(file_path: str, chunk_size: int = 500) -> list[Document]:
    """
    Load a text or PDF file and chunk it into documents.
    
    :param file_path: Path to the file (supports .txt, .pdf)
    :param chunk_size: Number of characters per chunk
    :return: List of Document objects
    """
    print(f"\nLoading file: {file_path}")
    
    # Check file extension
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.pdf':
        # Extract text from PDF
        try:
            import PyPDF2
            content = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                print(f"   PDF has {len(pdf_reader.pages)} pages")
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    content += page_text + "\n"
                    if (page_num + 1) % 10 == 0:
                        print(f"   Processed {page_num + 1} pages...")
        except ImportError:
            print("❌ PyPDF2 not installed. Install with: pip install PyPDF2")
            raise
    else:
        # Load as text file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    
    # Split into chunks
    chunks = []
    for i in range(0, len(content), chunk_size):
        chunk_text = content[i:i + chunk_size]
        if chunk_text.strip():  # Skip empty chunks
            chunks.append(
                Document(
                    id=f"chunk_{i // chunk_size}",
                    content=chunk_text,
                    meta={
                        "source": file_path,
                        "chunk_index": i // chunk_size,
                        "char_start": i,
                        "char_end": min(i + chunk_size, len(content)),
                    }
                )
            )
    
    print(f"✅ Created {len(chunks)} document chunks")
    return chunks


def generate_sample_documents(count: int) -> list[Document]:
    """
    Generate sample documents for testing.
    
    :param count: Number of documents to generate
    :return: List of Document objects
    """
    print(f"\n🔧 Generating {count} sample documents")
    
    sample_texts = [
        "Artificial intelligence is transforming the way we work and live.",
        "Machine learning models require large amounts of training data.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning networks can recognize patterns in complex data.",
        "Vector databases are optimized for similarity search operations.",
        "Embeddings represent text as high-dimensional vectors.",
        "Retrieval augmented generation combines search with language models.",
        "Semantic search finds documents based on meaning rather than keywords.",
        "Document stores provide efficient storage and retrieval of text data.",
        "Hybrid search combines keyword and semantic search for better results.",
    ]
    
    docs = []
    for i in range(count):
        text = sample_texts[i % len(sample_texts)]
        docs.append(
            Document(
                id=f"sample_{i}",
                content=f"{text} (Document {i})",
                meta={
                    "index": i,
                    "batch": i // 100,
                    "category": f"category_{i % 5}",
                }
            )
        )
    
    print(f"✅ Generated {len(docs)} documents")
    return docs


def generate_embeddings(documents: list[Document], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Generate embeddings for documents using sentence transformers.
    
    :param documents: List of documents
    :param model_name: Name of the sentence transformer model
    :return: Embedding dimension
    """
    print(f"\n🤖 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"🔄 Generating embeddings for {len(documents)} documents...")
    texts = [doc.content for doc in documents]
    
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Add embeddings to documents
    for doc, embedding in zip(documents, embeddings):
        doc.embedding = embedding.tolist()
    
    print(f"✅ Generated embeddings (dimension: {len(embeddings[0])})")
    
    return len(embeddings[0])


def benchmark_batch_insert(documents: list[Document], embedding_dim: int):
    """
    Benchmark batch insert mode (use_batch_insert=True).
    
    :param documents: Documents to insert
    :param embedding_dim: Embedding dimension
    :return: Time taken in seconds
    """
    print("\n" + "="*70)
    print("🚀 BATCH INSERT MODE (use_batch_insert=True)")
    print("="*70)
    
    use_ssl = os.getenv("DB2_SSL_ENABLED", "").lower() in {"1", "true", "yes"}
    port = int(os.getenv("DB2_SSL_PORT", "50001")) if use_ssl else int(os.getenv("DB2_PORT", "50000"))

    store = DB2DocumentStore(
        database=os.getenv("DB2_DATABASE", "TESTDB"),
        hostname=os.getenv("DB2_HOSTNAME"),
        port=port,
        username=Secret.from_env_var("DB2_USER"),
        password=Secret.from_env_var("DB2_PASSWORD"),
        table_name="benchmark_batch",
        embedding_dimension=embedding_dim,
        use_batch_insert=True,  # Enable batch insert
        batch_size=100,
        use_ssl=use_ssl,
        ssl_certificate=os.getenv("DB2_SSL_CERTIFICATE") or os.getenv("DB2_SSL_CERT_PATH"),
        recreate_table=True,
    )
    
    try:
        print(f"📊 Inserting {len(documents)} documents...")
        start_time = time.time()
        written = store.write_documents(documents)
        elapsed = time.time() - start_time
        
        print(f"✅ Inserted {written} documents")
        print(f"Time: {elapsed:.3f} seconds")
        print(f"Rate: {written / elapsed:.1f} docs/sec")
        
        # Verify count
        count = store.count_documents()
        print(f"✓ Verified: {count} documents in store")
        
        return elapsed
        
    finally:
        try:
            # store._drop_table_if_exists()
            pass 
        except Exception:
            pass


def benchmark_individual_insert(documents: list[Document], embedding_dim: int):
    """
    Benchmark individual insert mode (use_batch_insert=False).
    
    :param documents: Documents to insert
    :param embedding_dim: Embedding dimension
    :return: Time taken in seconds
    """
    print("\n" + "="*70)
    print("🐌 INDIVIDUAL INSERT MODE (use_batch_insert=False)")
    print("="*70)
    
    use_ssl = os.getenv("DB2_SSL_ENABLED", "").lower() in {"1", "true", "yes"}
    port = int(os.getenv("DB2_SSL_PORT", "50001")) if use_ssl else int(os.getenv("DB2_PORT", "50000"))

    store = DB2DocumentStore(
        database=os.getenv("DB2_DATABASE", "TESTDB"),
        hostname=os.getenv("DB2_HOSTNAME"),
        port=port,
        username=Secret.from_env_var("DB2_USER"),
        password=Secret.from_env_var("DB2_PASSWORD"),
        table_name="benchmark_individual",
        embedding_dimension=embedding_dim,
        use_batch_insert=False,  # Disable batch insert
        batch_size=100,
        use_ssl=use_ssl,
        ssl_certificate=os.getenv("DB2_SSL_CERTIFICATE") or os.getenv("DB2_SSL_CERT_PATH"),
        recreate_table=True,
    )
    
    try:
        print(f"📊 Inserting {len(documents)} documents...")
        start_time = time.time()
        written = store.write_documents(documents)
        elapsed = time.time() - start_time
        
        print(f"✅ Inserted {written} documents")
        print(f"Time: {elapsed:.3f} seconds")
        print(f"Rate: {written / elapsed:.1f} docs/sec")
        
        # Verify count
        count = store.count_documents()
        print(f"✓ Verified: {count} documents in store")
        
        return elapsed
        
    finally:
        try:
            store._drop_table_if_exists()
        except Exception:
            pass


def test_duplicate_handling(documents: list[Document], embedding_dim: int):
    """
    Test duplicate handling with batch insert.
    
    :param documents: Documents to use for testing
    :param embedding_dim: Embedding dimension
    """
    print("\n" + "="*70)
    print("DUPLICATE HANDLING TEST")
    print("="*70)
    
    # Use subset of documents for duplicate test
    test_docs = documents[:50] if len(documents) > 50 else documents
    
    use_ssl = os.getenv("DB2_SSL_ENABLED", "").lower() in {"1", "true", "yes"}
    port = int(os.getenv("DB2_SSL_PORT", "50001")) if use_ssl else int(os.getenv("DB2_PORT", "50000"))

    store = DB2DocumentStore(
        database=os.getenv("DB2_DATABASE", "TESTDB"),
        hostname=os.getenv("DB2_HOSTNAME"),
        port=port,
        username=Secret.from_env_var("DB2_USER"),
        password=Secret.from_env_var("DB2_PASSWORD"),
        table_name="benchmark_duplicates",
        embedding_dimension=embedding_dim,
        use_batch_insert=True,
        use_ssl=use_ssl,
        ssl_certificate=os.getenv("DB2_SSL_CERTIFICATE") or os.getenv("DB2_SSL_CERT_PATH"),
        recreate_table=True,
    )
    
    try:
        # Initial insert
        print(f"\n1️⃣ Initial insert of {len(test_docs)} documents...")
        written = store.write_documents(test_docs)
        print(f"   ✅ Inserted {written} documents")
        
        # Try to insert same documents with SKIP policy
        print(f"\n2️⃣ Re-inserting same documents with SKIP policy...")
        start_time = time.time()
        written = store.write_documents(test_docs, policy=DuplicatePolicy.SKIP)
        elapsed = time.time() - start_time
        print(f"   ✅ Skipped {len(test_docs) - written} duplicates")
        print(f"   Time: {elapsed:.3f} seconds")
        print(f"   Total documents: {store.count_documents()}")
        
        # Create mixed batch (some duplicates, some new)
        mixed_docs = test_docs[:25]  # First half are duplicates
        new_docs = [
            Document(
                id=f"new_{i}",
                content=f"New document {i}",
                embedding=test_docs[0].embedding,  # Reuse embedding
                meta={"type": "new"},
            )
            for i in range(25)
        ]
        mixed_docs.extend(new_docs)
        
        print(f"\n3️⃣ Inserting mixed batch (25 duplicates + 25 new) with SKIP policy...")
        start_time = time.time()
        written = store.write_documents(mixed_docs, policy=DuplicatePolicy.SKIP)
        elapsed = time.time() - start_time
        print(f"   ✅ Inserted {written} new documents (skipped 25 duplicates)")
        print(f"   Time: {elapsed:.3f} seconds")
        print(f"   Total documents: {store.count_documents()}")
        
        # Test OVERWRITE policy
        update_docs = [
            Document(
                id=test_docs[0].id,
                content="UPDATED CONTENT",
                embedding=test_docs[0].embedding,
                meta={"updated": True},
            )
        ]
        
        print(f"\n4️⃣ Updating document with OVERWRITE policy...")
        written = store.write_documents(update_docs, policy=DuplicatePolicy.OVERWRITE)
        print(f"   ✅ Updated {written} document")
        
        # Verify update
        updated = store.filter_documents(filters={"field": "id", "operator": "==", "value": test_docs[0].id})
        if updated:
            print(f"   ✓ Content updated: {updated[0].content[:50]}...")
        
    finally:
        try:
            store._drop_table_if_exists()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Benchmark DB2 bulk insert performance")
    parser.add_argument("--file", type=str, help="Path to text file to ingest")
    parser.add_argument("--generate", type=int, help="Generate N sample documents")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size for file splitting")
    parser.add_argument("--test-duplicates", action="store_true", help="Test duplicate handling")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Sentence transformer model name")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.file and not args.generate:
        parser.error("Either --file or --generate must be specified")
    
    print("\n" + "="*70)
    print("🔬 DB2 BULK INSERT PERFORMANCE BENCHMARK")
    print("="*70)
    
    # Load or generate documents
    if args.file:
        if not Path(args.file).exists():
            print(f"❌ Error: File not found: {args.file}")
            return
        documents = load_and_chunk_file(args.file, args.chunk_size)
    else:
        documents = generate_sample_documents(args.generate)
    
    # Generate embeddings with timing
    print(f"\n⏱️  Starting embedding generation...")
    embedding_start = time.time()
    embedding_dim = generate_embeddings(documents, args.model)
    embedding_time = time.time() - embedding_start
    print(f"⏱️  Embedding generation completed: {embedding_time:.3f}s ({len(documents) / embedding_time:.1f} docs/sec)")
    
    # Run benchmarks
    batch_time = benchmark_batch_insert(documents, embedding_dim)
    individual_time = benchmark_individual_insert(documents, embedding_dim)
    
    # Print comparison
    print("\n" + "="*70)
    print("📊 PERFORMANCE COMPARISON")
    print("="*70)
    print(f"Documents:        {len(documents)}")
    print(f"Embedding dim:    {embedding_dim}")
    
    print(f"\n⏱️  TIMING BREAKDOWN:")
    print(f"─" * 70)
    print(f"Embedding generation:  {embedding_time:.3f}s ({len(documents) / embedding_time:.1f} docs/sec)")
    print(f"Batch insert:          {batch_time:.3f}s ({len(documents) / batch_time:.1f} docs/sec)")
    print(f"Individual insert:     {individual_time:.3f}s ({len(documents) / individual_time:.1f} docs/sec)")
    
    print(f"\n📈 TOTAL TIME (Embedding + Insert):")
    print(f"─" * 70)
    print(f"Batch mode total:      {embedding_time + batch_time:.3f}s")
    print(f"Individual mode total: {embedding_time + individual_time:.3f}s")
    
    print(f"\n🚀 INSERT PERFORMANCE:")
    print(f"─" * 70)
    print(f"Speedup:               {individual_time / batch_time:.2f}x faster")
    print(f"Time saved:            {individual_time - batch_time:.3f}s ({(1 - batch_time/individual_time)*100:.1f}%)")
    
    # Test duplicate handling if requested
    if args.test_duplicates:
        test_duplicate_handling(documents, embedding_dim)
    
    print("\n" + "="*70)
    print("✅ BENCHMARK COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

# Made with Bob
