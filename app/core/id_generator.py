"""
ID Generator - Production-grade unique ID generation for multi-tenant system.

VECTORDB COMPATIBILITY:
- All IDs are safe for ChromaDB and Qdrant metadata filtering
- Special characters are escaped/removed
- Client IDs are normalized for consistent filtering
- Chunk IDs use UUID format for vector stores

ID Schema:
- client_id: Normalized tenant identifier (for VectorDB filtering)
- user_id: Unique user identifier (from auth system)
- session_id: Unique chat/conversation session
- doc_id: Unique document/file identifier  
- dataset_id: Unique dataset identifier (client:doc:sheet)
- chunk_id: Unique RAG chunk identifier (UUID format)
- query_id: Unique query identifier for audit trail

Format: {prefix}_{timestamp}_{random}
"""
import uuid
import time
import hashlib
import re
from typing import Optional, Tuple
from datetime import datetime


# =============================================================================
# VECTORDB SAFE ID GENERATION
# =============================================================================

# Characters allowed in VectorDB metadata (safe for both ChromaDB and Qdrant)
SAFE_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
UNSAFE_CHARS_PATTERN = re.compile(r'[^a-zA-Z0-9_\-\.]')


def normalize_client_id(client_id: str) -> str:
    """
    Normalize client_id for VectorDB compatibility.
    
    IMPORTANT: This ensures consistent filtering in ChromaDB and Qdrant.
    The same normalization MUST be applied at both ingestion and search time.
    
    Args:
        client_id: Raw client identifier
    
    Returns:
        Normalized client_id safe for VectorDB operations
    """
    if not client_id:
        return "default_client"
    
    # Lowercase, strip whitespace
    normalized = client_id.lower().strip()
    
    # Replace common separators with underscores
    normalized = normalized.replace(' ', '_')
    normalized = normalized.replace('-', '_')
    normalized = normalized.replace(':', '_')
    normalized = normalized.replace('/', '_')
    normalized = normalized.replace('\\', '_')
    
    # Remove any remaining unsafe characters
    normalized = UNSAFE_CHARS_PATTERN.sub('', normalized)
    
    # Collapse multiple underscores
    while '__' in normalized:
        normalized = normalized.replace('__', '_')
    
    # Remove leading/trailing underscores
    normalized = normalized.strip('_')
    
    return normalized or "default_client"


def generate_vectordb_safe_id(prefix: str = "", include_timestamp: bool = True) -> str:
    """
    Generate an ID that's safe for use in VectorDB metadata.
    
    Args:
        prefix: Optional prefix (e.g., 'doc', 'chunk')
        include_timestamp: Include timestamp for ordering
    
    Returns:
        VectorDB-safe unique ID
    """
    random_part = uuid.uuid4().hex[:12]
    
    if include_timestamp:
        timestamp = int(time.time() * 1000) % 10000000000  # 10 digits
        if prefix:
            return f"{prefix}_{timestamp}_{random_part}"
        return f"{timestamp}_{random_part}"
    else:
        if prefix:
            return f"{prefix}_{random_part}"
        return random_part


def generate_chunk_id_for_vectordb(
    client_id: str,
    doc_id: str,
    chunk_index: int
) -> Tuple[str, dict]:
    """
    Generate chunk ID and metadata for VectorDB storage.
    
    Returns both the point ID (for Qdrant) and metadata dict with
    properly normalized values for filtering.
    
    Args:
        client_id: Client/tenant identifier
        doc_id: Document identifier
        chunk_index: Chunk position in document
    
    Returns:
        Tuple of (point_id, metadata_dict)
    """
    # Normalize IDs for filtering
    safe_client = normalize_client_id(client_id)
    safe_doc = sanitize_id_component(doc_id)
    
    # Generate UUID-based point ID (required by Qdrant)
    point_id = str(uuid.uuid4())
    
    # Metadata with normalized values
    metadata = {
        "client_id": safe_client,
        "doc_id": safe_doc,
        "chunk_index": chunk_index,
        "chunk_key": f"{safe_client}_{safe_doc}_{chunk_index}",
        "created_at": get_iso_timestamp(),
    }
    
    return point_id, metadata


# =============================================================================
# CORE ID GENERATION FUNCTIONS
# =============================================================================

def generate_uuid() -> str:
    """Generate a random UUID4."""
    return str(uuid.uuid4())


def generate_short_id(prefix: str = "") -> str:
    """Generate a short unique ID with optional prefix."""
    timestamp = int(time.time() * 1000) % 100000000  # Last 8 digits of ms timestamp
    random_part = uuid.uuid4().hex[:8]
    if prefix:
        return f"{prefix}_{timestamp}_{random_part}"
    return f"{timestamp}_{random_part}"


def generate_user_id() -> str:
    """Generate unique user ID."""
    return generate_short_id("usr")


def generate_client_id(source: Optional[str] = None) -> str:
    """
    Generate or normalize a client ID.
    
    Args:
        source: Optional source string to derive client_id from
    
    Returns:
        Normalized, VectorDB-safe client ID
    """
    if source:
        return normalize_client_id(source)
    return generate_short_id("cli")


def generate_session_id(user_id: Optional[str] = None) -> str:
    """
    Generate unique session/chat ID.
    
    Args:
        user_id: Optional user ID to associate session with
    
    Returns:
        Unique session ID like 'ses_12345678_abc12345'
    """
    return generate_short_id("ses")


def generate_doc_id(user_id: str, filename: str) -> str:
    """
    Generate unique document ID.
    
    Args:
        user_id: User who uploaded the document
        filename: Original filename
    
    Returns:
        Unique document ID like 'doc_12345678_abc123'
    """
    # Include hash of filename for deduplication awareness
    file_hash = hashlib.md5(filename.encode()).hexdigest()[:6]
    timestamp = int(time.time() * 1000) % 100000000
    return f"doc_{timestamp}_{file_hash}"


def generate_dataset_id(client_id: str, doc_id: str, sheet_name: str) -> str:
    """
    Generate unique dataset ID with hierarchical structure.
    
    Uses normalized client_id for VectorDB compatibility.
    
    Args:
        client_id: Client/tenant ID (will be normalized)
        doc_id: Document ID
        sheet_name: Sheet or table name within document
    
    Returns:
        Hierarchical dataset ID like 'client:doc:sheet'
    """
    safe_client = normalize_client_id(client_id)
    safe_doc = sanitize_id_component(doc_id)
    safe_sheet = sanitize_id_component(sheet_name)
    
    return f"{safe_client}:{safe_doc}:{safe_sheet}"


def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
    """
    Generate unique chunk ID for RAG.
    
    Args:
        doc_id: Parent document ID
        chunk_index: Index of chunk within document
    
    Returns:
        Unique chunk ID
    """
    return f"{doc_id}_chunk_{chunk_index}"


def generate_query_id() -> str:
    """Generate unique query ID for audit trail."""
    return generate_short_id("qry")


def generate_request_id() -> str:
    """Generate unique request ID for request tracing."""
    return generate_short_id("req")


def generate_trace_id() -> str:
    """Generate unique trace ID for distributed tracing."""
    return generate_short_id("trace")


def generate_chat_id() -> str:
    """Generate unique chat/conversation ID."""
    return generate_short_id("chat")



# =============================================================================
# ID PARSING AND VALIDATION
# =============================================================================

def parse_dataset_id(dataset_id: str) -> dict:
    """
    Parse a dataset ID into its components.
    
    Args:
        dataset_id: Full dataset ID
    
    Returns:
        Dict with client_id, doc_id, sheet_name (if applicable)
    """
    parts = dataset_id.split(':')
    
    if len(parts) >= 3:
        return {
            "client_id": parts[0],
            "doc_id": parts[1],
            "sheet_name": ':'.join(parts[2:]),  # Handle colons in sheet name
            "full_id": dataset_id
        }
    elif len(parts) == 2:
        return {
            "client_id": parts[0],
            "doc_id": None,
            "sheet_name": parts[1],
            "full_id": dataset_id
        }
    else:
        return {
            "client_id": None,
            "doc_id": None,
            "sheet_name": parts[0] if parts else dataset_id,
            "full_id": dataset_id
        }


def validate_user_id(user_id: str) -> bool:
    """Validate user ID format."""
    if not user_id:
        return False
    # Allow alphanumeric, underscores, and hyphens
    return all(c.isalnum() or c in '_-' for c in user_id)


def validate_vectordb_id(id_value: str) -> bool:
    """
    Validate that an ID is safe for VectorDB operations.
    
    Args:
        id_value: ID to validate
    
    Returns:
        True if safe for ChromaDB and Qdrant
    """
    if not id_value:
        return False
    return bool(SAFE_ID_PATTERN.match(id_value))


def sanitize_id_component(value: str) -> str:
    """
    Sanitize a string for use in IDs.
    
    Ensures the result is safe for VectorDB metadata.
    """
    if not value:
        return "unnamed"
    
    # Remove special characters, lowercase, replace spaces with underscores
    sanitized = value.lower().strip()
    sanitized = sanitized.replace(' ', '_').replace('-', '_')
    sanitized = UNSAFE_CHARS_PATTERN.sub('', sanitized)
    
    # Collapse multiple underscores
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')
    
    sanitized = sanitized.strip('_')
    
    return sanitized or "unnamed"


def sanitize_metadata_value(value) -> str:
    """
    Sanitize a value for use in VectorDB metadata.
    
    ChromaDB and Qdrant only support primitive types in metadata.
    This function converts any value to a safe string.
    
    Args:
        value: Any value to sanitize
    
    Returns:
        String safe for VectorDB metadata
    """
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return str(value)
    return str(value)


# =============================================================================
# TIMESTAMP UTILITIES
# =============================================================================

def get_iso_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


def get_unix_timestamp() -> int:
    """Get current Unix timestamp in milliseconds."""
    return int(time.time() * 1000)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # VectorDB-safe ID functions
    "normalize_client_id",
    "generate_vectordb_safe_id",
    "generate_chunk_id_for_vectordb",
    "validate_vectordb_id",
    "sanitize_metadata_value",
    # Core ID functions
    "generate_uuid",
    "generate_short_id",
    "generate_user_id",
    "generate_client_id",
    "generate_session_id",
    "generate_doc_id",
    "generate_dataset_id",
    "generate_chunk_id",
    "generate_query_id",
    "generate_request_id",
    "generate_trace_id",
    "generate_chat_id",
    # Parsing and validation
    "parse_dataset_id",
    "validate_user_id",
    "sanitize_id_component",
    # Timestamps
    "get_iso_timestamp",
    "get_unix_timestamp",
]
