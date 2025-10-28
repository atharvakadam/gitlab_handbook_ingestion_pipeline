# GitLab Handbook Ingestion Pipeline

This pipeline processes the GitLab Handbook content, generates vector embeddings, and stores them in MongoDB for semantic search capabilities.

## Prerequisites

1. Python 3.8+
2. MongoDB (local or remote instance)
3. Git

## Setup

1. **Clone the GitLab Handbook repository**:
   ```bash
   git clone https://gitlab.com/gitlab-com/content-sites/handbook.git
   cd handbook
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Environment Variables

Create a `.env` file in the project root with the following variables:

```
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB=handbook_db
EMBEDDING_MODEL=intfloat/e5-base-v2
```

## Pipeline Components

### 1. `ingest_embed_handbook.py`

The main script that processes markdown files, generates embeddings, and stores them in MongoDB.

**Usage:**
```bash
python ingest_embed_handbook.py --handbook-root /path/to/handbook/repo --batch-size 32
```

**Key Features:**
- Processes markdown files from the GitLab Handbook
- Splits content into semantic chunks with configurable size/overlap
- Generates vector embeddings using Sentence Transformers
- Stores documents in MongoDB with rich metadata
- Supports resumable processing via content hashing
- Handles markdown elements like tables and images

### 2. `list_handbook_urls.py`

Helper script that maps local GitLab Handbook markdown files to their web and repository URLs.

**Usage:**
```bash
python list_handbook_urls.py --root /path/to/handbook/repo --sha main --output handbook_urls.csv
```

### 3. `qa_vector_search.py`

Script for performing semantic search on the embedded handbook content.

**Usage:**
```bash
python qa_vector_search.py --query "How does GitLab handle merge requests?"
```

### 4. `prefilter_cleanup.py`

Utility for cleaning up and prefiltering the handbook content before embedding.

## Data Model

Documents in MongoDB follow this structure:

```javascript
{
  doc_key: "path/to/doc#chunk_index",  // Unique identifier
  doc_id: "path/to/doc",               // Source document path
  chunk_index: 0,                      // Position of chunk in document
  title: "Document Title",            // Extracted from filename
  section: "engineering/development",  // Handbook section
  web_url: "https://...",              // Public URL to document
  repo_url: "https://...",             // Git URL to source file
  chunk_text: "...",                   // Text content (truncated)
  token_count: 123,                    // Approximate token count
  embedding: [0.1, 0.2, ...],         // Vector embedding
  embedding_model: "e5-base-v2",       // Model used for embedding
  sha: "abc123...",                    // Git commit SHA
  access_groups: ["all"],              // Access control
  updated_at: "2023-01-01T00:00:00Z",  // Processing timestamp
  content_hash: "sha256:...",          // Hash of original text
  source: "gitlab-handbook",           // Source identifier
  breadcrumbs: [],                     // Document hierarchy
  tags: []                             // Optional tags
}
```

## Running the Pipeline

1. **Clone and prepare the handbook repository**
   ```bash
   git clone https://gitlab.com/gitlab-com/content-sites/handbook.git
   cd handbook
   ```

2. **Set up the Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the ingestion pipeline**
   ```bash
   python ingest_embed_handbook.py --handbook-root /path/to/handbook/repo
   ```

4. **Verify the data**
   ```bash
   python qa_vector_search.py --query "How do I contribute to the handbook?"
   ```

## Configuration

### Environment Variables

- `MONGODB_URI`: MongoDB connection string (default: `mongodb://localhost:27017/`)
- `MONGODB_DB`: Database name (default: `handbook_db`)
- `EMBEDDING_MODEL`: Sentence Transformers model (default: `intfloat/e5-base-v2`)

### Command-line Arguments for `ingest_embed_handbook.py`

- `--handbook-root`: Path to the local GitLab Handbook repository (required)
- `--batch-size`: Number of documents to process in each batch (default: 32)
- `--max-tokens`: Maximum tokens per chunk (default: 550)
- `--overlap-tokens`: Token overlap between chunks (default: 80)
- `--collection`: MongoDB collection name (default: `handbook`)
- `--skip-embedding`: Skip embedding generation (for testing)
- `--resume`: Skip documents that already exist in the database

## Maintenance

- To update the handbook content, pull the latest changes and re-run the ingestion:
  ```bash
  cd /path/to/handbook
  git pull origin main
  cd /path/to/ingestion/pipeline
  python ingest_embed_handbook.py --handbook-root /path/to/handbook --resume
  ```

## Troubleshooting

- **Missing dependencies**: Ensure all required Python packages are installed from `requirements.txt`
- **MongoDB connection issues**: Verify MongoDB is running and accessible at the specified URI
- **Memory errors**: Reduce the `--batch-size` if processing large documents
- **Embedding model download**: The first run will download the specified model (may take time)

---

## License

This project is licensed under the **MIT License**.

**Author:** Atharva Kadam 

**Contact:** [atharvakadam@gmail.com](mailto:atharvakadam.dev@gmail.com)
