"""
STAGE 1 — Text Understanding
=============================
Parses natural language crime scene descriptions into structured semantics.

Input:  Raw text description (e.g., "Small bedroom. Broken window. Knife on table.")
Output: SceneSemantics dataclass with objects, attributes, relationships, scene type.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class SceneSemantics:
    """Structured output of text understanding stage."""
    raw_text: str
    scene_type: str = "unknown"
    objects: List[str] = field(default_factory=list)
    attributes: Dict[str, List[str]] = field(default_factory=dict)
    relationships: List[Tuple[str, str, str]] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "scene_type": self.scene_type,
            "objects": self.objects,
            "attributes": self.attributes,
            "relationships": [
                {"subject": s, "predicate": p, "object": o}
                for s, p, o in self.relationships
            ],
        }


# ── Scene type detection keywords ──────────────────────────────

SCENE_KEYWORDS = {
    "bedroom": ["bedroom", "bed", "pillow", "mattress", "nightstand", "closet"],
    "living_room": ["living room", "sofa", "couch", "television", "tv", "coffee table", "fireplace"],
    "kitchen": ["kitchen", "stove", "oven", "refrigerator", "fridge", "sink", "counter", "cabinet"],
    "bathroom": ["bathroom", "toilet", "shower", "bathtub", "basin", "mirror"],
    "hallway": ["hallway", "corridor", "passage"],
    "garage": ["garage", "car", "vehicle", "tool"],
    "basement": ["basement", "cellar"],
    "office": ["office", "desk", "computer", "monitor", "chair", "filing cabinet"],
    "alley": ["alley", "alleyway", "dumpster", "trash"],
    "parking_lot": ["parking lot", "parking", "car park"],
    "street": ["street", "road", "sidewalk", "curb", "pavement"],
    "warehouse": ["warehouse", "crate", "shelf", "rack"],
}

# ── Spatial/relational prepositions ────────────────────────────

SPATIAL_PREPS = [
    "on top of", "in front of", "next to", "on the left of", "on the right of",
    "behind", "under", "underneath", "beneath", "above", "over",
    "near", "beside", "close to", "against", "inside", "within",
    "on", "at", "by", "between", "around", "across from",
    "leaning against", "hanging from", "attached to", "lying on",
]

# Sort by length descending so longer phrases match first
SPATIAL_PREPS.sort(key=len, reverse=True)


class TextUnderstanding:
    """
    NLP parsing engine using spaCy for crime scene text understanding.
    """

    def __init__(self, nlp, scene_types: List[str] = None):
        """
        Args:
            nlp: spaCy Language model (en_core_web_sm).
            scene_types: Allowed scene type labels.
        """
        self.nlp = nlp
        self.scene_types = scene_types or list(SCENE_KEYWORDS.keys())

    def parse(self, text: str) -> SceneSemantics:
        """
        Parse a crime scene description into structured semantics.

        Steps:
          1. Sentence segmentation
          2. Scene type classification
          3. Object extraction
          4. Attribute extraction
          5. Relationship extraction
        """
        logger.info("=" * 50)
        logger.info("STAGE 1: Text Understanding")
        logger.info("=" * 50)
        logger.info(f"Input text: {text[:120]}...")

        result = SceneSemantics(raw_text=text)

        # Parse with spaCy
        doc = self.nlp(text)

        # 1. Sentence segmentation
        result.sentences = [sent.text.strip() for sent in doc.sents]
        logger.info(f"  Sentences: {len(result.sentences)}")

        # 2. Scene type classification
        result.scene_type = self._classify_scene(text)
        logger.info(f"  Scene type: {result.scene_type}")

        # 3. Object extraction
        result.objects = self._extract_objects(doc)
        logger.info(f"  Objects: {result.objects}")

        # 4. Attribute extraction
        result.attributes = self._extract_attributes(doc, result.objects)
        logger.info(f"  Attributes: {result.attributes}")

        # 5. Relationship extraction
        result.relationships = self._extract_relationships(doc, result.objects, text)
        logger.info(f"  Relationships: {result.relationships}")

        return result

    def _classify_scene(self, text: str) -> str:
        """Classify scene type based on keyword matching."""
        text_lower = text.lower()
        scores = {}
        for scene_type, keywords in SCENE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[scene_type] = score

        if scores:
            return max(scores, key=scores.get)
        return "unknown"

    def _extract_objects(self, doc) -> List[str]:
        """Extract physical objects from spaCy parse."""
        objects = set()

        for chunk in doc.noun_chunks:
            # Get the head noun
            head = chunk.root.text.lower()
            # Skip pronouns, abstract nouns, etc.
            if chunk.root.pos_ in ("NOUN", "PROPN"):
                objects.add(head)

        # Also add named entities that could be objects
        for ent in doc.ents:
            if ent.label_ in ("PRODUCT", "ORG", "GPE", "FAC"):
                objects.add(ent.text.lower())

        # Filter out very common non-object words
        stopwords = {"room", "scene", "area", "place", "thing", "side", "end", "type"}
        objects = [o for o in objects if o not in stopwords and len(o) > 1]

        return sorted(set(objects))

    def _extract_attributes(self, doc, objects: List[str]) -> Dict[str, List[str]]:
        """Extract adjective modifiers for detected objects."""
        attributes = {}

        for chunk in doc.noun_chunks:
            head = chunk.root.text.lower()
            if head in objects:
                adjs = []
                for token in chunk:
                    if token.pos_ == "ADJ" or token.dep_ == "amod":
                        adjs.append(token.text.lower())
                if adjs:
                    if head not in attributes:
                        attributes[head] = []
                    attributes[head].extend(adjs)

        # Deduplicate
        for obj in attributes:
            attributes[obj] = list(set(attributes[obj]))

        return attributes

    def _extract_relationships(self, doc, objects: List[str],
                                raw_text: str) -> List[Tuple[str, str, str]]:
        """
        Extract spatial relationships between objects.
        Uses both dependency parsing and pattern matching.
        """
        relationships = []
        seen = set()

        # Method 1: Dependency-based extraction
        for token in doc:
            if token.dep_ == "prep" and token.head.pos_ in ("NOUN", "PROPN"):
                subj = token.head.text.lower()
                prep = token.text.lower()
                # Find the object of the preposition
                for child in token.children:
                    if child.dep_ == "pobj":
                        obj = child.text.lower()
                        if subj in objects and obj in objects:
                            key = (subj, prep, obj)
                            if key not in seen:
                                relationships.append(key)
                                seen.add(key)

        # Method 2: Pattern-based extraction for spatial prepositions
        text_lower = raw_text.lower()
        for prep in SPATIAL_PREPS:
            # Pattern: <object> <prep> <object>
            for subj in objects:
                for obj in objects:
                    if subj == obj:
                        continue
                    pattern = rf'\b{re.escape(subj)}\b\s+{re.escape(prep)}\s+(?:the\s+|a\s+)?{re.escape(obj)}\b'
                    if re.search(pattern, text_lower):
                        key = (subj, prep, obj)
                        if key not in seen:
                            relationships.append(key)
                            seen.add(key)

        return relationships
