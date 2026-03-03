"""
STAGE 2 — Vocabulary Normalization
===================================
Standardizes object and relationship names using Visual Genome alias files.

Input:  SceneSemantics with raw object/relationship names
Output: SceneSemantics with canonicalized names
"""

import logging
from typing import Dict, List, Tuple
from copy import deepcopy

from src.stages.stage1_text_understanding import SceneSemantics

logger = logging.getLogger(__name__)


class VocabularyNormalizer:
    """
    Normalizes vocabulary using Visual Genome alias dictionaries.

    Alias format: {alias_str -> canonical_str}
      e.g., "couch" -> "sofa", "on top of" -> "on"
    """

    def __init__(self, object_aliases: Dict[str, str],
                 relationship_aliases: Dict[str, str]):
        self.object_aliases = object_aliases
        self.relationship_aliases = relationship_aliases
        logger.info(
            f"VocabularyNormalizer: {len(object_aliases)} object aliases, "
            f"{len(relationship_aliases)} relationship aliases"
        )

    def normalize_object(self, name: str) -> str:
        """Map an object name to its canonical form."""
        key = name.strip().lower()
        return self.object_aliases.get(key, key)

    def normalize_relationship(self, name: str) -> str:
        """Map a relationship name to its canonical form."""
        key = name.strip().lower()
        return self.relationship_aliases.get(key, key)

    def normalize_semantics(self, semantics: SceneSemantics) -> SceneSemantics:
        """
        Normalize all objects, attributes, and relationships in-place.

        Steps:
          1. Normalize object names
          2. Remap attribute keys to canonical names
          3. Normalize relationship subjects, predicates, and objects
          4. Deduplicate
        """
        logger.info("=" * 50)
        logger.info("STAGE 2: Vocabulary Normalization")
        logger.info("=" * 50)

        result = deepcopy(semantics)

        # 1. Normalize objects
        old_objects = result.objects
        obj_map = {}  # old_name -> canonical_name
        new_objects = []
        for obj in old_objects:
            canonical = self.normalize_object(obj)
            obj_map[obj] = canonical
            new_objects.append(canonical)

        result.objects = sorted(set(new_objects))

        log_changes = {k: v for k, v in obj_map.items() if k != v}
        if log_changes:
            logger.info(f"  Object normalization: {log_changes}")

        # 2. Normalize attribute keys
        new_attrs = {}
        for obj_name, attr_list in result.attributes.items():
            canonical = obj_map.get(obj_name, obj_name)
            if canonical not in new_attrs:
                new_attrs[canonical] = []
            new_attrs[canonical].extend(attr_list)
        # Deduplicate attribute values
        for k in new_attrs:
            new_attrs[k] = list(set(new_attrs[k]))
        result.attributes = new_attrs

        # 3. Normalize relationships
        new_rels: List[Tuple[str, str, str]] = []
        seen = set()
        for subj, pred, obj in result.relationships:
            norm_subj = obj_map.get(subj, self.normalize_object(subj))
            norm_pred = self.normalize_relationship(pred)
            norm_obj = obj_map.get(obj, self.normalize_object(obj))
            key = (norm_subj, norm_pred, norm_obj)
            if key not in seen:
                new_rels.append(key)
                seen.add(key)

        result.relationships = new_rels

        logger.info(f"  Final objects: {result.objects}")
        logger.info(f"  Final relationships: {result.relationships}")

        return result
