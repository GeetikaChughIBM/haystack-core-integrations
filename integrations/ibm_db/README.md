# ibm-db-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/ibm-db-haystack.svg)](https://pypi.org/project/ibm-db-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ibm-db-haystack.svg)](https://pypi.org/project/ibm-db-haystack)

Haystack DocumentStore backed by [IBM Db2 AI Vector Search](https://www.ibm.com/products/db2), available in Db2
11.5.9+ with the VECTOR data type enabled.

The integration exposes a **fully synchronous and asynchronous** document store surface, making it suitable
for both traditional blocking pipelines and async-first frameworks (FastAPI, asyncio). All async methods
delegate to the corresponding sync methods via `asyncio.to_thread` because the underlying `ibm_db_dbi`
driver is synchronous.

- [Integration page](https://haystack.deepset.ai/integrations/ibm-db-document-store)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/ibm_db/CHANGELOG.md)

---

## Quick start

```bash
pip install ibm-db-haystack
```

```python
import os
from haystack_integrations.document_stores.ibm_db import IBMDb2DocumentStore
from haystack_integrations.components.retrievers.ibm_db import IBMDb2EmbeddingRetriever
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy

os.environ["DB2_USERNAME"] = "db2inst1"
os.environ["DB2_PASSWORD"] = "s3cr3t"

store = IBMDb2DocumentStore(
    database="SAMPLEDB",
    hostname="db2.example.internal",
    port=50000,
    embedding_dim=768,
)

# Write documents
store.write_documents([
    Document(content="IBM Db2 supports native vector search.", meta={"source": "docs"}),
], policy=DuplicatePolicy.OVERWRITE)

# Async write (non-blocking)
import asyncio
asyncio.run(store.write_documents_async([
    Document(content="Async operations never block the event loop.", meta={"source": "blog"}),
]))

# Retrieve by embedding similarity
retriever = IBMDb2EmbeddingRetriever(document_store=store, top_k=5)
```

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, you need to have an IBM Db2 instance running.
You can start one using Docker:

```console
docker compose up -d
```
