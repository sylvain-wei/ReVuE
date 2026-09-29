"""Text rendering for student and teacher prefixes (Phase 1 Step 1 + Step 2).

Step 1 setup (answer-only privilege): completion-style templates loaded from .md files.
- template_for_student.md           -> student input (no privilege)
- template_for_teacher.md           -> teacher input, sees ground-truth answer

Step 2 setup (solution privilege): teacher additionally sees the full reference
solution (which already contains the boxed answer).
- template_for_teacher_solution.md  -> teacher input, sees reference solution

Both teacher templates and the student template end with ``Answer:`` so the
assistant continuation starts right after that marker; tokenizing prefix +
student_response yields a sequence where logits[prefix_len-1+k] predicts
response token k.

Template-mode detection (selected at construction time):
  - ``answer_only``    : teacher template contains {ground_truth_answer}
  - ``solution_only``  : teacher template contains {reference_solution}
                         (no separate {ground_truth_answer}; the solution
                         already includes \\boxed{...})

Each mode wires up the appropriate render method and the appropriate
validator branch; downstream callers can stay agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STUDENT_TEMPLATE = REPO_ROOT / "template_for_student.md"
DEFAULT_TEACHER_TEMPLATE = REPO_ROOT / "template_for_teacher.md"
DEFAULT_TEACHER_TEMPLATE_SOLUTION = REPO_ROOT / "template_for_teacher_solution.md"


def _safe_substitute(template: str, mapping: dict[str, str]) -> str:
    """Substitute {key} placeholders without being fooled by literal '{...}'
    inside problem text.

    Two-step strategy:
      1. Replace each placeholder with a unique sentinel that cannot appear
         in the original template or in user content
      2. Replace each sentinel with the real value
    """
    sentinels = {}
    out = template
    for i, key in enumerate(mapping.keys()):
        sentinel = f"\x00__OPD_FIELD_{i}__\x00"
        sentinels[sentinel] = mapping[key]
        out = out.replace("{" + key + "}", sentinel)
    for sentinel, value in sentinels.items():
        out = out.replace(sentinel, value)
    return out


class CompletionPrefixRenderer:
    """Render completion-style student/teacher prefixes from .md templates.

    Mode is auto-detected from the teacher template:
      - answer_only   : {problem}, {ground_truth_answer}
      - solution_only : {problem}, {reference_solution}

    Both templates end with ``Answer:`` so the model continues right after.
    """

    def __init__(
        self,
        student_template_path: Optional[Path] = None,
        teacher_template_path: Optional[Path] = None,
    ):
        student_path = Path(student_template_path) if student_template_path else DEFAULT_STUDENT_TEMPLATE
        teacher_path = Path(teacher_template_path) if teacher_template_path else DEFAULT_TEACHER_TEMPLATE

        if not student_path.is_file():
            raise FileNotFoundError(f"student template not found: {student_path}")
        if not teacher_path.is_file():
            raise FileNotFoundError(f"teacher template not found: {teacher_path}")

        self.student_template = student_path.read_text(encoding="utf-8")
        self.teacher_template = teacher_path.read_text(encoding="utf-8")
        self.student_template_path = student_path
        self.teacher_template_path = teacher_path

        # Sanity: student template must have {problem}.
        if "{problem}" not in self.student_template:
            raise ValueError(f"student template missing '{{problem}}' placeholder: {student_path}")

        # Sanity: teacher template must have {problem}.
        if "{problem}" not in self.teacher_template:
            raise ValueError(f"teacher template missing '{{problem}}' placeholder: {teacher_path}")

        # Mode detection.
        has_answer = "{ground_truth_answer}" in self.teacher_template
        has_solution = "{reference_solution}" in self.teacher_template
        if has_answer and has_solution:
            raise ValueError(
                f"teacher template has BOTH '{{ground_truth_answer}}' and "
                f"'{{reference_solution}}'; only one is supported: {teacher_path}"
            )
        if has_answer:
            self.template_mode = "answer_only"
        elif has_solution:
            self.template_mode = "solution_only"
        else:
            raise ValueError(
                f"teacher template missing privilege placeholder; expected one of "
                f"'{{ground_truth_answer}}' or '{{reference_solution}}': {teacher_path}"
            )

    def render_student_prefix(self, problem: str) -> str:
        return _safe_substitute(self.student_template, {"problem": problem})

    def render_teacher_prefix(self, problem: str, ground_truth_answer: str) -> str:
        """Step 1 path: privilege = ground-truth answer string only."""
        if self.template_mode != "answer_only":
            raise RuntimeError(
                f"render_teacher_prefix requires template_mode='answer_only', "
                f"got '{self.template_mode}' (template={self.teacher_template_path}). "
                f"Use render_teacher_prefix_with_solution for solution_only mode."
            )
        return _safe_substitute(
            self.teacher_template,
            {"problem": problem, "ground_truth_answer": ground_truth_answer},
        )

    def render_teacher_prefix_with_solution(
        self,
        problem: str,
        reference_solution: str,
    ) -> str:
        """Step 2 path: privilege = full reference solution (which contains
        the boxed final answer)."""
        if self.template_mode != "solution_only":
            raise RuntimeError(
                f"render_teacher_prefix_with_solution requires template_mode='solution_only', "
                f"got '{self.template_mode}' (template={self.teacher_template_path}). "
                f"Use render_teacher_prefix for answer_only mode."
            )
        return _safe_substitute(
            self.teacher_template,
            {"problem": problem, "reference_solution": reference_solution},
        )

    def validate_prefixes(
        self,
        problem: str,
        student_prefix: str,
        teacher_prefix: str,
        *,
        ground_truth_answer: Optional[str] = None,
        reference_solution: Optional[str] = None,
    ) -> dict[str, bool]:
        """Sanity checks on rendered prefixes. Mode-aware.

        For ``answer_only`` mode pass ``ground_truth_answer``.
        For ``solution_only`` mode pass ``reference_solution``.
        """
        checks: dict[str, bool] = {
            "student_has_problem": problem in student_prefix,
            "teacher_has_problem": problem in teacher_prefix,
            "student_ends_with_answer_marker": student_prefix.rstrip().endswith("Answer:"),
            "teacher_ends_with_answer_marker": teacher_prefix.rstrip().endswith("Answer:"),
            "no_unfilled_problem_placeholder": "{problem}" not in student_prefix
            and "{problem}" not in teacher_prefix,
            "teacher_longer_than_student": len(teacher_prefix) > len(student_prefix),
            "student_not_empty": len(student_prefix) > 0,
            "teacher_not_empty": len(teacher_prefix) > 0,
        }

        if self.template_mode == "answer_only":
            if ground_truth_answer is None:
                raise ValueError("ground_truth_answer required for answer_only validation")
            checks.update({
                "student_no_answer_leak": ground_truth_answer not in student_prefix,
                "teacher_has_privilege_block": "=== Ground-Truth Answer Begin ===" in teacher_prefix,
                "teacher_has_answer": ground_truth_answer in teacher_prefix,
                "no_unfilled_answer_placeholder": "{ground_truth_answer}" not in teacher_prefix,
            })
        elif self.template_mode == "solution_only":
            if reference_solution is None:
                raise ValueError("reference_solution required for solution_only validation")
            checks.update({
                # Solution itself can mention the problem text -- no leak check on student
                # other than "no privileged block markers leaked into student prefix".
                "student_no_solution_block_leak": "=== Reference Solution Begin ===" not in student_prefix,
                "teacher_has_privilege_block": "=== Reference Solution Begin ===" in teacher_prefix,
                "teacher_has_solution": reference_solution in teacher_prefix,
                "no_unfilled_solution_placeholder": "{reference_solution}" not in teacher_prefix,
            })
        else:
            raise RuntimeError(f"unknown template_mode: {self.template_mode}")

        return checks


# Backward-compatible alias so existing scripts that imported TextPrefixRenderer
# keep working.  The old API used (prompt, privileged_context); the new API uses
# (problem, ground_truth_answer) for answer_only mode and
# (problem, reference_solution) for solution_only mode.
class TextPrefixRenderer(CompletionPrefixRenderer):
    """Backward-compat shim. New code should use CompletionPrefixRenderer.

    In ``answer_only`` mode (the original Phase 1 Step 1 default) the old
    ``render_teacher_prefix(prompt, privileged_context)`` API is preserved
    by mapping ``privileged_context -> ground_truth_answer``.

    In ``solution_only`` mode (Phase 1 Step 2) ``render_teacher_prefix`` is
    repurposed to take the reference solution as the privileged_context; the
    underlying ``render_teacher_prefix_with_solution`` is used. This keeps the
    existing rollout/score scripts working with minimal changes -- they pass
    ``rollout["privileged_solution"]`` (or any string they have) as the
    second argument and the renderer routes correctly.
    """

    def __init__(self, use_system_prompt: bool = True, **kwargs):
        # ``use_system_prompt`` was the old toggle; it is ignored here because
        # the new templates fully define the system + instruction text.
        super().__init__(**kwargs)

    def render_student_prefix(self, prompt: str) -> str:  # type: ignore[override]
        return super().render_student_prefix(problem=prompt)

    def render_teacher_prefix(self, prompt: str, privileged_context: str) -> str:  # type: ignore[override]
        if self.template_mode == "answer_only":
            return super().render_teacher_prefix(
                problem=prompt, ground_truth_answer=privileged_context
            )
        elif self.template_mode == "solution_only":
            return super().render_teacher_prefix_with_solution(
                problem=prompt, reference_solution=privileged_context
            )
        raise RuntimeError(f"unknown template_mode: {self.template_mode}")

    def validate_prefixes(  # type: ignore[override]
        self,
        prompt: str,
        privileged_context: str,
        student_prefix: str,
        teacher_prefix: str,
    ) -> dict[str, bool]:
        if self.template_mode == "answer_only":
            return super().validate_prefixes(
                problem=prompt,
                student_prefix=student_prefix,
                teacher_prefix=teacher_prefix,
                ground_truth_answer=privileged_context,
            )
        elif self.template_mode == "solution_only":
            return super().validate_prefixes(
                problem=prompt,
                student_prefix=student_prefix,
                teacher_prefix=teacher_prefix,
                reference_solution=privileged_context,
            )
        raise RuntimeError(f"unknown template_mode: {self.template_mode}")


if __name__ == "__main__":
    # Self-test: both modes.
    print("=" * 60)
    print("MODE 1: answer_only (Step 1 default)")
    print("=" * 60)
    r1 = CompletionPrefixRenderer()  # default = template_for_teacher.md
    problem = "What is 2 + 2?"
    answer = "4"
    s = r1.render_student_prefix(problem)
    t = r1.render_teacher_prefix(problem, answer)

    print("=== STUDENT PREFIX ===")
    print(s)
    print(f"[len = {len(s)} chars]\n")
    print("=== TEACHER PREFIX (answer_only) ===")
    print(t)
    print(f"[len = {len(t)} chars]\n")
    print("=== VALIDATION ===")
    for k, v in r1.validate_prefixes(problem, s, t, ground_truth_answer=answer).items():
        print(f"  {'OK ' if v else 'BAD'} {k}: {v}")

    print()
    print("=" * 60)
    print("MODE 2: solution_only (Step 2)")
    print("=" * 60)
    r2 = CompletionPrefixRenderer(
        teacher_template_path=DEFAULT_TEACHER_TEMPLATE_SOLUTION
    )
    sol = "Step 1: 2+2=4. Therefore, the answer is \\boxed{4}."
    s2 = r2.render_student_prefix(problem)
    t2 = r2.render_teacher_prefix_with_solution(problem, sol)

    print("=== STUDENT PREFIX ===")
    print(s2)
    print(f"[len = {len(s2)} chars]\n")
    print("=== TEACHER PREFIX (solution_only) ===")
    print(t2)
    print(f"[len = {len(t2)} chars]\n")
    print("=== VALIDATION ===")
    for k, v in r2.validate_prefixes(problem, s2, t2, reference_solution=sol).items():
        print(f"  {'OK ' if v else 'BAD'} {k}: {v}")
