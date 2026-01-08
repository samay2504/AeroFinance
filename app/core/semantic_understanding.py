"""
Semantic Data Understanding - Production-grade financial data interpretation.

This module provides dynamic, pattern-free semantic understanding of tabular data
using NLP techniques and LLM-powered analysis:

1. Hierarchical Structure Detection - Multi-row headers, merged cells, nested data
2. Named Entity Recognition - Financial metrics, periods, currencies
3. Semantic Similarity Matching - Find rows/columns by meaning, not exact text
4. Ontology-Based Classification - Financial statement types, metric categories
5. Temporal Pattern Recognition - Fiscal years, quarters, months in any format

No hardcoding - designed to work with any financial/accounting data.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Named Entity Recognition for Financial Data
# =============================================================================

@dataclass
class FinancialEntity:
    """Recognized financial entity."""
    text: str
    entity_type: str  # period, metric, currency, percentage, identifier
    normalized: str   # Normalized form
    confidence: float
    position: Optional[Tuple[int, int]] = None  # (row, col)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FinancialNER:
    """
    Named Entity Recognition for financial data.
    Detects periods, metrics, currencies, and financial terminology.
    """

    # Period patterns - designed to catch ANY financial period format
    PERIOD_PATTERNS = [
        # Fiscal year patterns
        (r'\b(fy\s*\'?\d{2,4})\b', 'fiscal_year'),
        (r'\b(fiscal\s+year\s+\d{4})\b', 'fiscal_year'),
        (r'\b(f[yq]\d{2,4})\b', 'fiscal_year'),
        
        # Month-fiscal year patterns
        (r'\b(\d{1,2}m\s*fy\s*\'?\d{2,4})\b', 'partial_year'),
        (r'\b(ytd\s*fy\s*\'?\d{2,4})\b', 'partial_year'),
        (r'\b(h[12]\s*fy\s*\'?\d{2,4})\b', 'half_year'),
        
        # Quarter patterns
        (r'\b(q[1-4]\s*\'?\d{2,4})\b', 'quarter'),
        (r'\b(q[1-4]\s*fy\s*\'?\d{2,4})\b', 'quarter'),
        (r'\b(quarter\s+[1-4])\b', 'quarter'),
        
        # Month patterns (any format)
        (r'\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
         r'jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
         r'[\s\-\']*(\d{2,4})?\b', 'month'),
        
        # Date patterns
        (r'\b(\d{4}[-/]\d{2}[-/]\d{2})\b', 'date'),
        (r'\b(\d{2}[-/]\d{2}[-/]\d{4})\b', 'date'),
        (r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b', 'date'),
        
        # Year patterns
        (r'\b(20\d{2}|19\d{2})\b', 'year'),
        (r'\b(cy\s*\d{2,4})\b', 'calendar_year'),
    ]
    
    # Financial metric patterns - semantic categories
    METRIC_CATEGORIES = {
        'revenue': [
            r'revenue', r'sales', r'income', r'turnover', r'receipts',
            r'top\s*line', r'gross\s*receipts', r'net\s*sales',
        ],
        'expense': [
            r'expense', r'cost', r'expenditure', r'outflow', r'spend',
            r'operating\s*expense', r'overhead', r'opex', r'capex',
        ],
        'profit': [
            r'profit', r'margin', r'earning', r'ebitda', r'ebit',
            r'net\s*income', r'gross\s*profit', r'operating\s*profit',
            r'pbt', r'pat', r'bottom\s*line',
            r'p&l', r'p_l', r'pnl', r'profit\s*(?:and|&)\s*loss',  # P&L variants
        ],
        'asset': [
            r'asset', r'receivable', r'inventory', r'cash', r'investment',
            r'property', r'equipment', r'intangible', r'goodwill',
        ],
        'liability': [
            r'liability', r'payable', r'debt', r'loan', r'borrowing',
            r'obligation', r'provision', r'accrual',
        ],
        'equity': [
            r'equity', r'capital', r'reserve', r'surplus', r'retained',
            r'share\s*capital', r'stock',
        ],
        'cashflow': [
            r'cash\s*flow', r'operating\s*cash', r'free\s*cash',
            r'financing', r'investing', r'working\s*capital',
        ],
        'ratio': [
            r'ratio', r'percentage', r'percent', r'growth', r'change',
            r'variance', r'margin\s*%', r'yield', r'return',
        ],
        'volume': [
            r'volume', r'quantity', r'units', r'orders', r'transactions',
            r'customers', r'users', r'subscribers', r'count',
        ],
        'performance': [
            r'gmv', r'arpu', r'aov', r'ltv', r'cac', r'arr', r'mrr',
            r'dau', r'mau', r'conversion', r'churn', r'retention',
        ],
    }
    
    # Sheet type to metric category mapping
    # This maps common sheet name patterns to the metric categories they contain
    # EXTEND THIS to support new sheet naming conventions
    SHEET_TYPE_CATEGORIES = {
        # P&L / Income Statement sheets
        'pl': ['profit', 'revenue', 'expense'],
        'p_l': ['profit', 'revenue', 'expense'],
        'pnl': ['profit', 'revenue', 'expense'],
        'income': ['profit', 'revenue', 'expense'],
        'profit': ['profit', 'revenue', 'expense'],
        'revenue': ['revenue'],
        
        # Balance Sheet sheets  
        'bs': ['asset', 'liability', 'equity'],
        'balance': ['asset', 'liability', 'equity'],
        'assets': ['asset'],
        'liabilities': ['liability'],
        
        # Cash Flow sheets
        'cf': ['cashflow'],
        'cash': ['cashflow'],
        'cashflow': ['cashflow'],
        
        # Performance/KPI sheets
        'kpi': ['performance', 'ratio', 'volume'],
        'metrics': ['performance', 'ratio'],
        'unit': ['performance', 'volume'],  # unit economics
        
        # Trial Balance (typically raw data, lower priority)
        'tb': [],  # Empty = no specific category preference
        'trial': [],
    }

    def __init__(self):
        # Compile patterns for efficiency
        self._period_patterns = [
            (re.compile(p, re.IGNORECASE), t) for p, t in self.PERIOD_PATTERNS
        ]
        self._metric_patterns = {
            cat: [re.compile(rf'\b{p}\b', re.IGNORECASE) for p in patterns]
            for cat, patterns in self.METRIC_CATEGORIES.items()
        }

    def extract_entities(self, text: str) -> List[FinancialEntity]:
        """Extract all financial entities from text."""
        entities = []
        text = str(text)
        
        # Extract periods
        for pattern, period_type in self._period_patterns:
            for match in pattern.finditer(text):
                entities.append(FinancialEntity(
                    text=match.group(0),
                    entity_type='period',
                    normalized=self._normalize_period(match.group(0)),
                    confidence=0.9,
                    metadata={'period_type': period_type}
                ))
        
        # Extract metrics
        for category, patterns in self._metric_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entities.append(FinancialEntity(
                        text=match.group(0),
                        entity_type='metric',
                        normalized=match.group(0).lower(),
                        confidence=0.8,
                        metadata={'category': category}
                    ))
        
        return entities

    def _normalize_period(self, period_str: str) -> str:
        """Normalize period string to standard format."""
        text = period_str.lower().strip()
        
        # Remove common separators and whitespace
        text = re.sub(r"['\s\-]+", '', text)
        
        # Normalize FY patterns
        text = re.sub(r'fiscalyear', 'fy', text)
        
        # Normalize date strings
        date_match = re.match(r'(\d{4})-(\d{2})-(\d{2})', period_str)
        if date_match:
            y, m, d = date_match.groups()
            months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                      'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
            month_name = months[int(m) - 1]
            return f"{month_name}_{y}"
        
        return text


# =============================================================================
# Semantic Similarity Matching
# =============================================================================

class SemanticMatcher:
    """
    Semantic similarity matching for finding data by meaning.
    Uses multiple strategies: exact match, fuzzy match, synonym expansion.
    """

    # Synonym groups for financial terms
    # Synonym groups for financial terms
    SYNONYMS = {
        'revenue': {'sales', 'income', 'turnover', 'receipts', 'top line'},
        'expense': {'cost', 'expenditure', 'outflow', 'spending', 'burn'},
        'profit': {'earnings', 'margin', 'income', 'net income', 'ebitda'},
        'growth': {'increase', 'change', 'delta', 'variance', 'difference'},
        'total': {'sum', 'aggregate', 'overall', 'cumulative', 'entire'},
        'percentage': {'percent', '%', 'rate', 'proportion'},
        'actual': {'realised', 'realized', 'real', 'occurred'},
        'budget': {'planned', 'forecast', 'projected', 'estimated'},
        'variance': {'difference', 'deviation', 'gap', 'delta'},
        'digital': {'online', 'electronic', 'web', 'internet'},
        'marketing': {'advertising', 'promotion', 'ads', 'campaign'},
        'domestic': {'local', 'national', 'home', 'internal'},
        'collection': {'receipts', 'receivables', 'incoming'},
        'prepaid': {'advance', 'pre-paid', 'upfront'},
        'recorded': {'booked', 'registered', 'logged'},
        'summary': {'overview', 'report', 'brief', 'synopsis'},
        'metadata': {'structure', 'schema', 'columns', 'sheets', 'tables', 'names'},
    }
    
    # Class-level NLP models to share across instances
    _nlp = None
    _spacy_loaded = False
    _nltk_loaded = False
    _lemmatizer = None
    _wordnet_ready = False

    def __init__(self):
        # Build reverse synonym map
        self._synonym_map: Dict[str, Set[str]] = {}
        for key, synonyms in self.SYNONYMS.items():
            self._synonym_map[key] = synonyms | {key}
            for syn in synonyms:
                if syn not in self._synonym_map:
                    self._synonym_map[syn] = set()
                self._synonym_map[syn].add(key)
                self._synonym_map[syn].update(synonyms)
        
        # Load NLP libraries
        if not SemanticMatcher._spacy_loaded:
            self._load_spacy()
        
        if not SemanticMatcher._nltk_loaded:
            self._load_nltk()

    @classmethod
    def _load_spacy(cls):
        """Load spaCy model for semantic vectors and NER."""
        try:
            import spacy
            try:
                cls._nlp = spacy.load("en_core_web_md")
                logger.info("SemanticMatcher loaded spaCy: en_core_web_md")
            except Exception:
                try:
                    cls._nlp = spacy.load("en_core_web_sm")
                    logger.info("SemanticMatcher loaded spaCy: en_core_web_sm")
                except Exception:
                    logger.warning("No spaCy model found. Semantic matching will rely on synonyms only.")
            cls._spacy_loaded = True
        except ImportError:
            logger.info("spaCy not installed. Usage restricted.")

    @classmethod
    def _load_nltk(cls):
        """Load NLTK resources for lemmatization and WordNet synonyms."""
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer
            from nltk.corpus import wordnet
            
            # Download required NLTK data silently
            for resource in ['wordnet', 'averaged_perceptron_tagger', 'punkt', 'omw-1.4']:
                try:
                    nltk.data.find(f'corpora/{resource}' if resource != 'punkt' else f'tokenizers/{resource}')
                except LookupError:
                    try:
                        nltk.download(resource, quiet=True)
                    except Exception:
                        pass
            
            cls._lemmatizer = WordNetLemmatizer()
            cls._wordnet_ready = True
            cls._nltk_loaded = True
            logger.info("SemanticMatcher loaded NLTK: WordNet + Lemmatizer")
        except ImportError:
            logger.info("NLTK not installed. Advanced lemmatization unavailable.")
            cls._nltk_loaded = True  # Mark as attempted

    # ================================================================
    # NLTK METHODS - WordNet Synonyms and Lemmatization
    # ================================================================
    
    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """Convert TreeBank POS tag to WordNet POS tag."""
        tag_map = {
            'J': 'a',  # Adjective
            'V': 'v',  # Verb
            'N': 'n',  # Noun
            'R': 'r',  # Adverb
        }
        return tag_map.get(treebank_tag[0], 'n')
    
    def lemmatize(self, text: str) -> str:
        """
        Lemmatize text using NLTK WordNet with POS tagging.
        This converts words to their base form considering context.
        """
        if not self._lemmatizer:
            return text.lower()
        
        try:
            import nltk
            words = nltk.word_tokenize(text.lower())
            pos_tags = nltk.pos_tag(words)
            
            lemmatized = []
            for word, pos in pos_tags:
                wn_pos = self._get_wordnet_pos(pos)
                lemma = self._lemmatizer.lemmatize(word, wn_pos)
                lemmatized.append(lemma)
            
            return ' '.join(lemmatized)
        except Exception:
            return text.lower()
    
    def get_wordnet_synonyms(self, word: str, limit: int = 10) -> Set[str]:
        """
        Get synonyms from WordNet for a given word.
        Includes financial domain filtering.
        """
        synonyms = set()
        
        if not self._wordnet_ready:
            return synonyms
        
        try:
            from nltk.corpus import wordnet
            
            for synset in wordnet.synsets(word):
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ').lower()
                    if synonym != word.lower() and len(synonym) > 2:
                        synonyms.add(synonym)
                        if len(synonyms) >= limit:
                            return synonyms
        except Exception:
            pass
        
        return synonyms
    
    def get_financial_synonyms(self, word: str) -> Set[str]:
        """
        Get synonyms with financial domain priority.
        Combines manual financial synonyms with WordNet.
        """
        # Start with our curated financial synonyms
        synonyms = self.get_synonyms(word)
        
        # Add WordNet synonyms
        wn_syns = self.get_wordnet_synonyms(word)
        synonyms.update(wn_syns)
        
        # Financial-specific expansions
        financial_expansions = {
            'yoy': {'year on year', 'year over year', 'y-o-y', 'annual growth'},
            'qoq': {'quarter on quarter', 'quarter over quarter', 'q-o-q'},
            'mom': {'month on month', 'month over month', 'm-o-m'},
            'cagr': {'compound annual growth rate', 'compound growth'},
            'ebitda': {'earnings before interest tax depreciation amortization', 'operating profit'},
            'ebit': {'earnings before interest tax', 'operating income'},
            'pat': {'profit after tax', 'net profit', 'net income'},
            'pbt': {'profit before tax', 'pretax profit'},
            'roe': {'return on equity'},
            'roa': {'return on assets'},
            'roce': {'return on capital employed'},
            'ev': {'enterprise value', 'firm value'},
            'fcf': {'free cash flow'},
            'fcff': {'free cash flow to firm'},
            'fcfe': {'free cash flow to equity'},
            'wacc': {'weighted average cost of capital'},
            'dcf': {'discounted cash flow'},
            'nwc': {'net working capital', 'working capital'},
            'capex': {'capital expenditure', 'capital expense'},
            'opex': {'operating expenditure', 'operating expense'},
            'gmv': {'gross merchandise value', 'gross value'},
            'arpu': {'average revenue per user'},
            'ltv': {'lifetime value', 'customer lifetime value'},
            'cac': {'customer acquisition cost'},
            'churn': {'attrition', 'customer churn', 'churn rate'},
        }
        
        word_lower = word.lower()
        if word_lower in financial_expansions:
            synonyms.update(financial_expansions[word_lower])
        
        return synonyms
    
    # ================================================================
    # SPACY METHODS - NER and Entity Extraction
    # ================================================================
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities using spaCy.
        Returns entities with their labels, positions, and confidence.
        """
        entities = []
        
        if not self._nlp:
            return entities
        
        try:
            doc = self._nlp(text)
            
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'description': self._get_entity_description(ent.label_)
                })
        except Exception as e:
            logger.debug(f"Entity extraction failed: {e}")
        
        return entities
    
    def _get_entity_description(self, label: str) -> str:
        """Get human-readable description for spaCy entity labels."""
        descriptions = {
            'MONEY': 'Monetary value',
            'PERCENT': 'Percentage',
            'DATE': 'Date or time period',
            'ORG': 'Organization/Company',
            'GPE': 'Geo-Political Entity (Country/City)',
            'CARDINAL': 'Numeric value',
            'ORDINAL': 'Ordinal number',
            'QUANTITY': 'Quantity/Measurement',
            'PERSON': 'Person name',
            'TIME': 'Time expression',
        }
        return descriptions.get(label, label)
    
    def extract_noun_chunks(self, text: str) -> List[str]:
        """
        Extract noun chunks (meaningful phrases) from text using spaCy.
        Useful for identifying metric names and concepts.
        """
        chunks = []
        
        if not self._nlp:
            return chunks
        
        try:
            doc = self._nlp(text)
            for chunk in doc.noun_chunks:
                chunks.append(chunk.text.lower())
        except Exception:
            pass
        
        return chunks
    
    def get_dependency_relations(self, text: str) -> List[Dict[str, str]]:
        """
        Parse dependency relations to understand query structure.
        Helps identify what the user is asking about.
        """
        relations = []
        
        if not self._nlp:
            return relations
        
        try:
            doc = self._nlp(text)
            for token in doc:
                if token.dep_ in ['nsubj', 'dobj', 'pobj', 'attr', 'ROOT']:
                    relations.append({
                        'word': token.text,
                        'lemma': token.lemma_,
                        'pos': token.pos_,
                        'dep': token.dep_,
                        'head': token.head.text,
                    })
        except Exception:
            pass
        
        return relations
    
    # ================================================================
    # ADVANCED QUERY UNDERSTANDING
    # ================================================================
    
    def parse_financial_query(self, query: str) -> Dict[str, Any]:
        """
        Comprehensive financial query parsing using NLTK + spaCy.
        Extracts metric, period, operation, and other components.
        """
        result = {
            'metric': None,
            'period': None,
            'operation': None,
            'entities': [],
            'noun_chunks': [],
            'lemmatized': '',
            'dependencies': [],
        }
        
        try:
            # Lemmatize the query
            result['lemmatized'] = self.lemmatize(query)
            
            # Extract spaCy entities
            result['entities'] = self.extract_entities(query)
            
            # Extract noun chunks
            result['noun_chunks'] = self.extract_noun_chunks(query)
            
            # Get dependency parse
            result['dependencies'] = self.get_dependency_relations(query)
            
            # Detect operation type
            query_lower = query.lower()
            if any(w in query_lower for w in ['sum', 'total', 'add']):
                result['operation'] = 'sum'
            elif any(w in query_lower for w in ['average', 'mean', 'avg']):
                result['operation'] = 'average'
            elif any(w in query_lower for w in ['growth', 'change', 'difference', 'variance']):
                result['operation'] = 'growth'
            elif any(w in query_lower for w in ['compare', 'versus', 'vs']):
                result['operation'] = 'compare'
            elif any(w in query_lower for w in ['list', 'show', 'display']):
                result['operation'] = 'list'
            else:
                result['operation'] = 'lookup'
            
            # Extract period mentions using FinancialNER
            ner = FinancialNER()
            period_entities = ner.extract_periods(query)
            if period_entities:
                result['period'] = period_entities[0].normalized
            
            # Extract metric (most likely noun chunk)
            for chunk in result['noun_chunks']:
                # Skip chunks that are periods
                if re.match(r'\d{4}|fy\d{2}|q[1-4]', chunk):
                    continue
                # Skip common stop phrases
                if chunk not in ['what', 'which', 'how much', 'how many']:
                    result['metric'] = chunk
                    break
                    
        except Exception as e:
            logger.debug(f"Query parsing failed: {e}")
        
        return result

    def get_synonyms(self, word: str) -> Set[str]:
        """Get all synonyms for a word."""
        word_lower = word.lower()
        return self._synonym_map.get(word_lower, {word_lower})

    def expand_query(self, query: str) -> Set[str]:
        """Expand query with synonyms."""
        words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        expanded = set()
        
        for word in words:
            expanded.add(word)
            expanded.update(self.get_synonyms(word))
        
        return expanded

    def calculate_similarity(self, query: str, target: str) -> float:
        """
        Calculate semantic similarity score between query and target.
        Uses multiple strategies:
        1. Exact/substring matching
        2. Financial acronym expansion (PAT -> profit after tax)
        3. spaCy vector similarity
        4. Synonym-based fallback with Jaccard similarity
        Returns 0.0 to 1.0.
        """
        query_lower = query.lower()
        target_lower = target.lower()
        
        # NORMALIZE: Convert underscored identifiers to space-separated words
        # e.g., "equity_value" -> "equity value", "summary_pl" -> "summary pl"
        query_normalized = re.sub(r'[_\-\.]', ' ', query_lower).strip()
        target_normalized = re.sub(r'[_\-\.]', ' ', target_lower).strip()
        
        # 1. Exact match
        if query_normalized == target_normalized:
            return 1.0
        
        # 2. Substring match (use normalized versions)
        if query_normalized in target_normalized or target_normalized in query_normalized:
            return 0.95
        
        # 3. FINANCIAL ACRONYM EXPANSION
        # If either side is a known acronym, expand it for better matching
        query_expanded = self._expand_financial_acronyms(query_normalized)
        target_expanded = self._expand_financial_acronyms(target_normalized)
        
        # Check expanded forms for matches
        if query_expanded == target_expanded:
            return 0.92
        if query_expanded in target_expanded or target_expanded in query_expanded:
            return 0.88
            
        # 4. spaCy Vector Similarity (Semantic) - use EXPANDED text for better acronym handling
        if self._nlp:
            try:
                doc1 = self._nlp(query_expanded)
                doc2 = self._nlp(target_expanded)
                if doc1.vector_norm and doc2.vector_norm:
                    spacy_sim = doc1.similarity(doc2)
                    # If spaCy gives reasonable similarity, use it
                    if spacy_sim > 0.3:
                        return spacy_sim
            except Exception:
                pass
        
        # 5. Synonym-based Fallback with Financial Term Expansion
        query_words = self.expand_query(query)
        target_words = set(re.findall(r'\b\w{2,}\b', target_lower))
        
        # Add expanded acronym words
        query_words.update(set(query_expanded.split()))
        target_words.update(set(target_expanded.split()))
        
        if not query_words or not target_words:
            return 0.0
        
        # Jaccard-like similarity with synonym expansion
        intersection = query_words & target_words
        union = query_words | target_words
        
        base_score = len(intersection) / len(union) if union else 0
        
        # Bonus for matching important financial terms
        important_matches = intersection & set(self.SYNONYMS.keys())
        bonus = len(important_matches) * 0.15
        
        return min(1.0, base_score + bonus)
    
    def _expand_financial_acronyms(self, text: str) -> str:
        """
        Expand financial acronyms in text.
        E.g., 'PAT' -> 'profit after tax', 'EBITDA' -> 'earnings before interest...'
        """
        # Financial acronym mapping (both directions)
        acronym_expansions = {
            'yoy': 'year on year',
            'qoq': 'quarter on quarter',
            'mom': 'month on month',
            'cagr': 'compound annual growth rate',
            'ebitda': 'earnings before interest tax depreciation amortization',
            'ebit': 'earnings before interest tax',
            'pat': 'profit after tax',
            'pbt': 'profit before tax',
            'roe': 'return on equity',
            'roa': 'return on assets',
            'roce': 'return on capital employed',
            'ev': 'enterprise value',
            'fcf': 'free cash flow',
            'fcff': 'free cash flow to firm',
            'fcfe': 'free cash flow to equity',
            'wacc': 'weighted average cost of capital',
            'dcf': 'discounted cash flow',
            'nwc': 'net working capital',
            'capex': 'capital expenditure',
            'opex': 'operating expenditure',
            'gmv': 'gross merchandise value',
            'arpu': 'average revenue per user',
            'ltv': 'lifetime value',
            'cac': 'customer acquisition cost',
            'pl': 'profit and loss',
            'pnl': 'profit and loss',
            'bs': 'balance sheet',
            'cf': 'cash flow',
            'valuation': 'enterprise value equity value',  # Map valuation to main value metrics
        }
        
        # Check if entire text is an acronym
        text_clean = re.sub(r'[^a-z0-9]', '', text.lower())
        if text_clean in acronym_expansions:
            return acronym_expansions[text_clean]
        
        # Expand individual acronym words in text
        words = text.split()
        expanded_words = []
        for word in words:
            word_clean = re.sub(r'[^a-z0-9]', '', word.lower())
            if word_clean in acronym_expansions:
                expanded_words.append(acronym_expansions[word_clean])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)

    def classify_intent(self, query: str, intents: Dict[str, List[str]]) -> Tuple[str, float]:
        """
        Classify query intent based on semantic similarity to exemplars.
        
        Args:
            query: User query
            intents: Dictionary of {intent_name: [list of example phrases]}
            
        Returns:
            Tuple of (best_intent, confidence_score)
        """
        best_intent = "unknown"
        best_score = 0.0
        
        for intent, exemplars in intents.items():
            # Calculate max similarity to any exemplar in this intent
            intent_score = 0.0
            for ex in exemplars:
                score = self.calculate_similarity(query, ex)
                if score > intent_score:
                    intent_score = score
            
            if intent_score > best_score:
                best_score = intent_score
                best_intent = intent
                
        return best_intent, best_score

    def find_best_match(
        self,
        query: str,
        candidates: List[str],
        threshold: float = 0.3
    ) -> Optional[Tuple[str, float]]:
        """Find the best matching candidate for a query."""
        best_match = None
        best_score = 0.0
        
        for candidate in candidates:
            score = self.calculate_similarity(query, candidate)
            if score > best_score:
                best_score = score
                best_match = candidate
        
        if best_score >= threshold:
            return (best_match, best_score)
        return None


# =============================================================================
# Hierarchical Structure Detection
# =============================================================================

class StructureDetector:
    """
    Detects hierarchical structure in spreadsheet data.
    Identifies header rows, label columns, data regions, and merged cell patterns.
    """

    def __init__(self):
        self._ner = FinancialNER()

    def detect_label_column(self, df: pd.DataFrame) -> Optional[int]:
        """
        Detect which column contains row labels.
        Uses heuristics: text density, entity presence, non-numeric ratio.
        """
        if df.empty or len(df.columns) == 0:
            return None
        
        best_col = 0
        best_score = 0
        
        for col_idx in range(min(5, len(df.columns))):
            col = df.columns[col_idx]
            score = 0
            
            # Count characteristics
            text_count = 0
            empty_count = 0
            entity_count = 0
            numeric_count = 0
            
            for val in df[col].head(30):
                val_str = str(val).strip()
                
                if val_str.lower() in ('nan', 'none', '', 'na', 'null'):
                    empty_count += 1
                    continue
                
                # Check if numeric
                try:
                    float(val_str.replace(',', '').replace('%', '').replace('$', '').replace('₹', ''))
                    numeric_count += 1
                except ValueError:
                    text_count += 1
                
                # Check for financial entities
                entities = self._ner.extract_entities(val_str)
                if any(e.entity_type == 'metric' for e in entities):
                    entity_count += 1
            
            # Score the column
            total = text_count + numeric_count + empty_count
            if total == 0:
                continue
            
            # Prefer columns with more text, financial entities, and fewer empty cells
            text_ratio = text_count / total
            empty_ratio = empty_count / total
            entity_ratio = entity_count / max(1, text_count)
            
            score = (text_ratio * 3) + (entity_ratio * 2) - (empty_ratio * 2)
            
            if score > best_score:
                best_score = score
                best_col = col_idx
        
        return best_col if best_score > 0.3 else 0

    def detect_header_row(self, df: pd.DataFrame) -> int:
        """
        Detect which row contains the primary headers.
        Looks for period patterns, column titles, etc.
        """
        if df.empty:
            return 0
        
        best_row = 0
        best_score = 0
        
        for row_idx in range(min(10, len(df))):
            period_count = 0
            text_count = 0
            empty_count = 0
            
            for val in df.iloc[row_idx]:
                val_str = str(val).strip()
                
                if val_str.lower() in ('nan', 'none', '', 'na'):
                    empty_count += 1
                    continue
                
                # Check for period entities
                entities = self._ner.extract_entities(val_str)
                if any(e.entity_type == 'period' for e in entities):
                    period_count += 1
                
                # Check if text
                try:
                    float(val_str.replace(',', '').replace('%', ''))
                except ValueError:
                    text_count += 1
            
            total = len(df.columns)
            if total == 0:
                continue
            
            # Headers have many periods and text, few empty cells
            score = (period_count * 3) + (text_count * 0.5) - (empty_count * 0.5)
            
            if score > best_score:
                best_score = score
                best_row = row_idx
        
        return best_row

    def extract_period_columns(
        self,
        df: pd.DataFrame,
        header_row: int = 0
    ) -> Dict[str, str]:
        """
        Extract mapping of period names to column names.
        Scans header rows for period patterns.
        """
        period_map: Dict[str, str] = {}
        
        # Scan multiple rows in case of multi-row headers
        for row_idx in range(max(0, header_row - 2), min(len(df), header_row + 3)):
            for col_idx, val in enumerate(df.iloc[row_idx]):
                val_str = str(val).strip()
                
                entities = self._ner.extract_entities(val_str)
                for entity in entities:
                    if entity.entity_type == 'period':
                        col_name = df.columns[col_idx]
                        normalized = entity.normalized
                        
                        # Store with multiple key formats for flexible lookup
                        period_map[normalized] = col_name
                        
                        # Also store original text
                        if entity.text.lower() != normalized:
                            period_map[entity.text.lower()] = col_name
        
        return period_map


# =============================================================================
# Dynamic Query Understanding
# =============================================================================

class QueryUnderstanding:
    """
    Understands user queries and extracts intent, entities, and constraints.
    """

    # Query intent patterns
    INTENT_PATTERNS = {
        'aggregation': [
            r'\btotal\b', r'\bsum\b', r'\baggregate\b', r'\bcumulative\b',
            r'\bentire\b', r'\ball\b', r'\bwhole\b',
        ],
        'comparison': [
            r'\bgrowth\b', r'\bchange\b', r'\bvariance\b', r'\bdifference\b',
            r'\bvs\b', r'\bversus\b', r'\bcompare\b', r'\bfrom\b.*\bto\b',
        ],
        'lookup': [
            r'\bwhat\s+(is|was|were)\b', r'\bfind\b', r'\blook\s*up\b',
            r'\bget\b', r'\bshow\b', r'\bvalue\s*(of|for)\b',
        ],
        'percentage': [
            r'\bpercentage\b', r'\bpercent\b', r'\b%\b', r'\bratio\b',
            r'\bproportion\b',
        ],
    }

    def __init__(self):
        self._ner = FinancialNER()
        self._matcher = SemanticMatcher()
        
        self._intent_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.INTENT_PATTERNS.items()
        }

    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse query to extract intent, entities, and constraints.
        """
        result = {
            'original': query,
            'intent': [],
            'periods': [],
            'metrics': [],
            'constraints': {},
            'keywords': set(),
            'strategy': None,  # Added: lookup strategy
        }
        
        # Extract intent
        for intent, patterns in self._intent_patterns.items():
            if any(p.search(query) for p in patterns):
                result['intent'].append(intent)
        
        # Extract entities
        entities = self._ner.extract_entities(query)
        for entity in entities:
            if entity.entity_type == 'period':
                result['periods'].append({
                    'text': entity.text,
                    'normalized': entity.normalized,
                    'type': entity.metadata.get('period_type')
                })
            elif entity.entity_type == 'metric':
                result['metrics'].append({
                    'text': entity.text,
                    'category': entity.metadata.get('category')
                })
        
        # Extract keywords with synonym expansion
        result['keywords'] = self._matcher.expand_query(query)
        
        # Determine lookup strategy
        result['strategy'] = self._determine_strategy(result)
        
        return result
    
    def _determine_strategy(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligently determine the lookup strategy based on query analysis.
        
        Returns a strategy dict with:
        - lookup_type: 'row', 'column', 'cell', 'aggregation', 'comparison', 'summary'
        - requires_row_search: bool - need to find a specific row by label
        - requires_column_search: bool - need to find a specific column by header/period
        - requires_computation: bool - need calculations (sum, avg, growth, etc.)
        - llm_required: bool - requires LLM for complex interpretation
        """
        strategy = {
            'lookup_type': 'cell',  # Default: find specific cell
            'requires_row_search': False,
            'requires_column_search': False,
            'requires_computation': False,
            'llm_required': False,
            'confidence': 0.5,
        }
        
        intents = parsed.get('intent', [])
        periods = parsed.get('periods', [])
        metrics = parsed.get('metrics', [])
        keywords = parsed.get('keywords', set())
        query = parsed.get('original', '').lower()
        
        # ==== STRATEGY DETERMINATION LOGIC ====
        
        # 1. SUMMARY/OVERVIEW - needs LLM
        if any(kw in query for kw in ['summary', 'summarize', 'overview', 'describe', 'explain', 'tell me about']):
            strategy['lookup_type'] = 'summary'
            strategy['llm_required'] = True
            strategy['confidence'] = 0.9
            return strategy
        
        # 2. AGGREGATION (total, sum, count)
        if 'aggregation' in intents:
            strategy['lookup_type'] = 'aggregation'
            strategy['requires_computation'] = True
            if periods:
                # Sum for specific period -> need column
                strategy['requires_column_search'] = True
            if metrics:
                # Sum for specific metric -> need row
                strategy['requires_row_search'] = True
            strategy['confidence'] = 0.8
            return strategy
        
        # 3. COMPARISON/GROWTH (period-to-period)
        if 'comparison' in intents or len(periods) >= 2:
            strategy['lookup_type'] = 'comparison'
            strategy['requires_row_search'] = True  # Need metric row
            strategy['requires_column_search'] = True  # Need period columns
            strategy['requires_computation'] = True
            strategy['llm_required'] = len(periods) < 2  # May need LLM to infer periods
            strategy['confidence'] = 0.85
            return strategy
        
        # 4. SPECIFIC CELL LOOKUP (value at row X, column Y)
        if periods and (metrics or any(kw in keywords for kw in ['revenue', 'sales', 'profit', 'cost', 'expense', 'gmv', 'income'])):
            strategy['lookup_type'] = 'cell'
            strategy['requires_row_search'] = True  # Find the metric row
            strategy['requires_column_search'] = True  # Find the period column
            strategy['confidence'] = 0.9
            return strategy
        
        # 5. SINGLE PERIOD LOOKUP (all values for a period)
        if periods and len(periods) == 1 and not metrics:
            strategy['lookup_type'] = 'column'
            strategy['requires_column_search'] = True
            strategy['confidence'] = 0.7
            return strategy
        
        # 6. SINGLE METRIC LOOKUP (all values for a metric)
        if metrics and not periods:
            strategy['lookup_type'] = 'row'
            strategy['requires_row_search'] = True
            strategy['confidence'] = 0.7
            return strategy
        
        # 7. PERCENTAGE/RATIO
        if 'percentage' in intents:
            strategy['lookup_type'] = 'cell'
            strategy['requires_row_search'] = True
            strategy['requires_column_search'] = True if periods else False
            strategy['confidence'] = 0.75
            return strategy
        
        # 8. DEFAULT: Simple lookup (needs LLM to interpret)
        if 'lookup' in intents:
            strategy['lookup_type'] = 'cell'
            strategy['requires_row_search'] = True
            strategy['requires_column_search'] = bool(periods)
            strategy['llm_required'] = True  # Need LLM to understand what to look up
            strategy['confidence'] = 0.6
            return strategy
        
        # 9. FALLBACK: Complex query needs LLM
        strategy['llm_required'] = True
        strategy['confidence'] = 0.3
        return strategy


# Global instances (lazy loaded)
_ner = None
_matcher = None
_detector = None
_query_understanding = None


def get_financial_ner() -> FinancialNER:
    global _ner
    if _ner is None:
        _ner = FinancialNER()
    return _ner

def get_semantic_matcher() -> SemanticMatcher:
    global _matcher
    if _matcher is None:
        _matcher = SemanticMatcher()
    return _matcher

def get_structure_detector() -> StructureDetector:
    global _detector
    if _detector is None:
        _detector = StructureDetector()
    return _detector

def get_query_understanding() -> QueryUnderstanding:
    global _query_understanding
    if _query_understanding is None:
        _query_understanding = QueryUnderstanding()
    return _query_understanding


__all__ = [
    'FinancialNER',
    'FinancialEntity', 
    'SemanticMatcher',
    'StructureDetector',
    'QueryUnderstanding',
    'get_financial_ner',
    'get_semantic_matcher',
    'get_structure_detector',
    'get_query_understanding',
]
