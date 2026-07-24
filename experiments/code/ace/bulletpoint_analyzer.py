"""
BulletpointAnalyzer for AppWorld ACE.

Embedding-based similarity grouping + optional LLM merge of similar playbook
bullets. Adapted from ace-main's BulletpointAnalyzer to AppWorld's playbook
format (`[id] content`) and LiteLLMGenerator.
"""

from __future__ import annotations

import re
from typing import Any

from .playbook import format_playbook_line, parse_playbook_line

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    DEDUP_AVAILABLE = True
except ImportError:
    DEDUP_AVAILABLE = False
    np = None  # type: ignore[assignment]
    print(
        "Warning: sentence-transformers or faiss not available for bulletpoint analysis. "
        "Install with: pip install sentence-transformers faiss-cpu"
    )


class BulletpointAnalyzer:
    """Deduplicate / LLM-merge similar AppWorld playbook bullets."""

    def __init__(
        self,
        llm_generator: Any,
        embedding_model_name: str = "all-mpnet-base-v2",
    ):
        self.llm_generator = llm_generator
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        if not DEDUP_AVAILABLE:
            print("⚠️  Bulletpoint analyzer initialized but dependencies not available")

    def _load_embedding_model(self) -> None:
        if self.embedding_model is None and DEDUP_AVAILABLE:
            print(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

    def _parse_playbook(
        self, playbook: str
    ) -> tuple[list[str], list[dict[str, Any]], dict[int, int]]:
        lines = playbook.strip().split("\n")
        bullets: list[dict[str, Any]] = []
        bullet_line_mapping: dict[int, int] = {}
        for line_idx, line in enumerate(lines):
            if line.strip().startswith("#"):
                continue
            parsed = parse_playbook_line(line)
            if not parsed:
                continue
            bullet = {
                "id": parsed["id"],
                "helpful": int(parsed.get("helpful", 0) or 0),
                "harmful": int(parsed.get("harmful", 0) or 0),
                "content": (parsed.get("content") or "").strip(),
                "line_number": line_idx + 1,
                "original_line": line,
            }
            bullet_line_mapping[len(bullets)] = line_idx
            bullets.append(bullet)
        return lines, bullets, bullet_line_mapping

    def _compute_embeddings(self, bullets: list[dict[str, Any]]):
        if not DEDUP_AVAILABLE:
            raise RuntimeError("Cannot compute embeddings without sentence-transformers")
        self._load_embedding_model()
        contents = [b["content"] for b in bullets]
        embeddings = self.embedding_model.encode(
            contents, convert_to_numpy=True, show_progress_bar=False
        )
        faiss.normalize_L2(embeddings)
        return embeddings

    def _find_similar_groups(
        self,
        bullets: list[dict[str, Any]],
        embeddings,
        threshold: float,
    ) -> list[dict[str, Any]]:
        similarity_matrix = np.dot(embeddings, embeddings.T)
        duplicate_groups: list[dict[str, Any]] = []
        visited: set[int] = set()
        for i in range(len(bullets)):
            if i in visited:
                continue
            similar_indices = [
                j
                for j in range(i + 1, len(bullets))
                if similarity_matrix[i, j] >= threshold
            ]
            if similar_indices:
                group = [i] + similar_indices
                duplicate_groups.append(
                    {"indices": group, "bullets": [bullets[idx] for idx in group]}
                )
                visited.update(group)
        return duplicate_groups

    def _merge_bullets_with_llm(
        self, bullets_group: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        if len(bullets_group) == 1:
            return bullets_group[0]

        bullets_text = "\n".join(
            f"{i + 1}. [{b['id']}] {b['content']}" for i, b in enumerate(bullets_group)
        )
        base_id = bullets_group[0]["id"]
        prompt = f"""You are merging similar playbook bulletpoints into a single, comprehensive entry.

Given these similar bulletpoints:
{bullets_text}

Merge them into ONE bulletpoint that captures all important information while removing redundancy.

Requirements:
1. Keep the ID from the first entry: [{base_id}]
2. Combine the content to be comprehensive but concise
3. Output ONLY in this format: [{base_id}] [merged content]

Do NOT include any explanation, just output the merged bulletpoint."""

        try:
            response = self.llm_generator.generate(
                messages=[{"role": "user", "content": prompt}],
                extra_log_fields={"ace_role": "bulletpoint_analyzer"},
            )
            merged_content = (response.get("content") or "").strip()
            # Prefer AppWorld format: [id] content
            match = re.match(r"\[([^\]]+)\]\s+(.*)", merged_content, re.DOTALL)
            if match:
                bullet_id, content = match.groups()
                content = content.strip()
                # Strip optional helpful/harmful :: prefix if model echoes finance format
                content = re.sub(
                    r"^helpful=\d+\s+harmful=\d+\s*::\s*", "", content
                ).strip()
                return {
                    "id": bullet_id,
                    "helpful": 0,
                    "harmful": 0,
                    "content": content,
                    "original_line": format_playbook_line(bullet_id, 0, 0, content),
                    "is_merged": True,
                    "original_count": len(bullets_group),
                }
            print("⚠️  Failed to parse merged bullet, keeping first bullet from group")
            return bullets_group[0]
        except Exception as e:
            print(f"⚠️  Error merging bullets: {e}, keeping first bullet from group")
            return bullets_group[0]

    def analyze(
        self,
        playbook: str,
        threshold: float = 0.90,
        merge: bool = True,
    ) -> str:
        if not DEDUP_AVAILABLE:
            print("⚠️  Skipping bulletpoint analysis (dependencies not available)")
            return playbook

        original_lines, bullets, bullet_line_mapping = self._parse_playbook(playbook)
        if not bullets:
            return playbook

        print(f"Analyzing {len(bullets)} bulletpoints (threshold={threshold})...")
        embeddings = self._compute_embeddings(bullets)
        duplicate_groups = self._find_similar_groups(bullets, embeddings, threshold)
        if not duplicate_groups:
            print(f"No similar bulletpoints found at threshold {threshold}")
            return playbook

        print(f"Found {len(duplicate_groups)} groups of similar bulletpoints")
        merge_mapping: dict[int, dict[str, Any]] = {}
        processed_indices: set[int] = set()

        if merge:
            for group_idx, group in enumerate(duplicate_groups):
                indices = group["indices"]
                group_bullets = group["bullets"]
                print(f"  Merging group {group_idx + 1}: {len(group_bullets)} bullets -> 1")
                merged_bullet = self._merge_bullets_with_llm(group_bullets)
                if merged_bullet:
                    merge_mapping[indices[0]] = merged_bullet
                    processed_indices.update(indices)
        else:
            for group in duplicate_groups:
                processed_indices.update(group["indices"][1:])

        output_lines: list[str] = []
        for line_idx, original_line in enumerate(original_lines):
            current_bullet_idx = None
            for bi, mapped_line in bullet_line_mapping.items():
                if mapped_line == line_idx:
                    current_bullet_idx = bi
                    break

            if current_bullet_idx is not None:
                if current_bullet_idx in merge_mapping:
                    output_lines.append(merge_mapping[current_bullet_idx]["original_line"])
                elif current_bullet_idx in processed_indices:
                    continue
                else:
                    output_lines.append(original_line)
            else:
                output_lines.append(original_line)

        final_bullet_count = len(bullets) - len(processed_indices) + len(merge_mapping)
        removed_count = len(bullets) - final_bullet_count
        print(
            f"✓ Bulletpoint analysis complete: {len(bullets)} -> {final_bullet_count} "
            f"({removed_count} bullets merged/removed)"
        )
        return "\n".join(output_lines)
