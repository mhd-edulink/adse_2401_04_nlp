"""
=================================================================================================
Python script to demonstrate ABSA (Aspect Based Sentiment Analysis)
=================================================================================================

WHAT IS ABSA?
-------------
Standard sentiment analysis assigns ONE sentiment to an entire sentence:
    "The battery is great but the screen is awful."  →  MIXED

ABSA assigns sentiment PER ASPECT (topic/feature):
    battery  →  POSITIVE
    screen   →  NEGATIVE

This gives far richer, more actionable insights — especially useful for
product reviews, customer feedback, and survey analysis.

HOW THIS DEMO WORKS
--------------------
  1. ASPECT EXTRACTION  — rule-based keyword matching across 9 product
                          categories (battery, screen, camera, etc.)
  2. CLAUSE SPLITTING   — the sentence is split at conjunctions (but, and,
                          however, …) so that each clause is scored in
                          isolation, preventing sentiments from bleeding
                          across aspects.
  3. SENTIMENT SCORING  — VADER (Valence Aware Dictionary and sEntiment
                          Reasoner) scores each clause.  VADER is purpose-
                          built for short, informal text and requires NO
                          model download or GPU.
"""


# ------------------------------------------------------------------------------------------
# 0. Import the required modules
# ------------------------------------------------------------------------------------------
from __future__ import annotations # Ensure this is the 1st import to avoid errors
import re, sys, textwrap
from dataclasses import dataclass, field
from typing import Optional

# ------------------------------------------------------------------------------------------
# 1. Check dependancies
# ------------------------------------------------------------------------------------------
def _require(package: str, install_cmd: str) -> None:
    import importlib.util
    if importlib.util.find_spec(package) is None:
        print(f"\n[ERROR] Required package {package} not found"
              f"\n Install it with {install_cmd}\n")

        sys.exit(1)

_require("vaderSentiment", "pip install vaderSentiment")

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ------------------------------------------------------------------------------------------
# 2. Aspect taxonomy
# ------------------------------------------------------------------------------------------

# Each key is the canonical aspect name shown in the output.
# Each value is a list of surface form keywords (singular, plurarl, synonyms).
# Matching is case-sensitive and whole-word only (regext \b boundaries)
ASPECT_TAXONOMY: dict[str, list[str]] = {
    "battery life":   ["battery life", "battery", "charge", "charging"],
    "screen":         ["screen", "display", "monitor", "resolution", "brightness"],
    "camera":         ["camera", "cameras", "photo", "photos", "lens", "zoom"],
    "performance":    ["performance", "speed", "processor", "lag", "fast", "slow", "snappy"],
    "price":          ["price", "cost", "expensive", "cheap", "affordable", "value"],
    "service":        ["service", "services", "support", "staff", "customer service"],
    "build quality":  ["build quality", "build", "design", "materials", "durability", "sturdy"],
    "storage":        ["storage", "memory", "space", "capacity"],
    "audio":          ["audio", "sound", "speaker", "speakers", "headphone", "headphones", "bass"],
}

# VADER compound-score thresholds (industry standard values)
POSITIVE_THRESHOLD =  0.05
NEGATIVE_THRESHOLD = -0.05


# ----------------------------------------------------------------------------------------------
# 3. Data classes
# ----------------------------------------------------------------------------------------------
@dataclass
class AspectResult:
    """Sentiment prediction for one aspect within a sentence."""
    aspect: str
    sentiment: str  # "POSITIVE" | "NEUTRAL" | "NEGATIVE"
    score: float  # VADER compound score in [-1, +1]
    confidence: str  # "HIGH" | "MEDIUM" | "LOW"
    source: str  # the clause that the aspect was found in


@dataclass
class SentenceResult:
    """Full ABSA output for one input sentence."""
    text: str
    aspects: list[AspectResult] = field(default_factory=list)
    overall_label: Optional[str] = None  # overall VADER sentiment
    overall_score: Optional[float] = None


# -------------------------------------------------------------------------
# 4. Core ABSA Engine
# -------------------------------------------------------------------------
class ABSAAnalyser:
    def __init__(
            self,
            taxonomy: dict[str, list[str]] | None = None,
            positive_threshold: float = POSITIVE_THRESHOLD,
            negative_threshold=NEGATIVE_THRESHOLD,
    ) -> None:
        self._vader = SentimentIntensityAnalyzer()
        self._taxonomy = taxonomy or ASPECT_TAXONOMY
        self._pos_thr = positive_threshold
        self._neg_thr = negative_threshold
        # pre compiler patterns: longest keywords first to avoid partial matches. eg battery life must be checked before battery
        self._patterns: dict[str, list[re.Pattern[str]]] = {}
        for aspect, keywords in self._taxonomy.items():
            sorted_kw = sorted(keywords, key=len, reverse=True)
            self._patterns[aspect] = [
                re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in sorted_kw
            ]

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def analyze(self, text: str) -> SentenceResult:
        result = SentenceResult(text=text)
        # Overall sentence-level sentiment
        overall_scores = self._vader.polarity_scores(text)
        result.overall_score = overall_scores["compound"]
        result.overall_label = self._label(result.overall_score)
        # Split into classes so sentiments don't bleed across aspects
        clauses = self._split_clauses(text)
        # match aspects to clauses and score match
        seen_aspects: set[str] = set()
        for clause in clauses:
            for aspect, patterns in self._patterns.items():
                if aspect in seen_aspects:
                    continue
                for pattern in patterns:
                    if pattern.search(clause):
                        scores = self._vader.polarity_scores(clause)
                        compound = scores["compound"]
                        sentiment = self._label(compound)
                        confidence = self._confidence(compound)
                        result.aspects.append(
                            AspectResult(
                                aspect=aspect,
                                sentiment=sentiment,
                                score=compound,
                                confidence=confidence,
                                source=clause.strip(),
                            )
                        )
                        seen_aspects.add(aspect)
                        break  # stop checking other keywords for this aspect
        return result

    def analyze_batch(self, texts: list[str]) -> list[SentenceResult]:
        return [self.analyze(t) for t in texts]

    # ---------------------------------------------------------------------
    # Private Helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _split_clauses(text: str) -> list[str]:
        # Primary split on contrastive/additive connectors
        parts = re.split(
            r"\s*(?:but|however|although|though|yet|while|;|,)\s*",
            text,
            flags=re.IGNORECASE,
        )
        # Secondary split: "and the/a/an …" — new noun phrase starting
        clauses: list[str] = []
        for part in parts:
            sub = re.split(
                r"\s+and\s+(?=(?:the|a|an|this|its|my|their)\s)",
                part,
                flags=re.IGNORECASE,
            )
            clauses.extend(sub)
        return [c.strip() for c in clauses if c.strip()]

    def _label(self, compound: float) -> str:
        if compound >= self._pos_thr:
            return "POSITIVE"
        if compound <= self._neg_thr:
            return "NEGATIVE"
        return "NEUTRAL"

    @staticmethod
    def _confidence(compound: float) -> str:
        abs_score = abs(compound)
        if abs_score >= 0.5:
            return "HIGH"
        if abs_score <= 0.2:
            return "MEDIUM"
        return "LOW"


# -------------------------------------------------------------------------
# 5. Display Helpers
# -------------------------------------------------------------------------
_SENTIMENT_SYMBOLS = {"POSITIVE": "✓", "NEGATIVE": "✗", "NEUTRAL": "~"}
_SENTIMENT_COLORS = {
    "POSITIVE": "\033[32m",  # green
    "NEGATIVE": "\033[31m",  # red
    "NEUTRAL": "\033[33m",  # yellow
}
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _colorize(text: str, color_code: str) -> str:
    return f"{color_code}{text}{_RESET}"


def print_result(result: SentenceResult, show_clause: bool = False) -> None:
    width = 72
    print("\n" + "-" * width)
    print(f"{_BOLD}TEXT:{_RESET}{result.text}")
    if result.overall_label:
        sym = _SENTIMENT_SYMBOLS[result.overall_label]
        color = _SENTIMENT_COLORS[result.overall_label]
        label = _colorize(f"{sym} {result.overall_label}", color)
        print(f"Overall sentiment: {label}"
              f"(score: {result.overall_score:+.3f})")
    if not result.aspects:
        print("\n No known product aspects detected")
        return
    print(f"\n  {'Aspect':<16}  {'Sentiment':<10}  {'Conf.':<8}  {'Score':>7}")
    print(f"  {'─' * 15}  {'─' * 9}  {'─' * 7}  {'─' * 7}")
    for r in result.aspects:
        sym = _SENTIMENT_SYMBOLS[r.sentiment]
        color = _SENTIMENT_COLORS[r.sentiment]
        label = _colorize(f"{sym} {r.sentiment:<9}", color)
        print(f"  {r.aspect:<16}  {label}  {r.confidence:<8}  {r.score:+.3f}")
        if show_clause:
            wrapped = textwrap.fill(
                f'"{r.source}"',
                width=width - 22,
                subsequent_indent=" " * 22,
            )
            print(f"  {'':16}  ↳ clause: {wrapped}")


def print_summary_table(results: list[SentenceResult]) -> None:
    width = 72
    print("\n" + "-" * width)
    print(f"{_BOLD} BATCH SUMMARY:{_RESET}")
    print("-" * width)
    for n, result in enumerate(results, start=1):
        short = (result.text[:55] + "…") if len(result.text) > 55 else result.text
        print(f"\n [{n}] {short}")
        if result.aspects:
            parts = []
            for r in result.aspects:
                sym = _SENTIMENT_SYMBOLS[r.sentiment]
                color = _SENTIMENT_COLORS[r.sentiment]
                parts.append(_colorize(text=f"{sym} {r.aspect}", color_code=color))
            print("     " + " | ".join(parts))
        else:
            print("⚠️ No aspects found")
    print("-" * width)


# -------------------------------------------------------------------------
# 5. Display Helpers
# -------------------------------------------------------------------------
DEMO_SENTENCES = [
    # Clear multi-aspect contrasts
    "The battery life is excellent but the screen is disappointing.",
    "Camera quality is stunning and the performance is incredibly fast.",
    "Battery life is awful and the price is way too high.",
    # Mixed service + storage
    "Customer service was amazing, though the storage is far too limited.",
    # Audio vs screen
    "The display is absolutely gorgeous but the audio sounds terrible.",
    # Mild / neutral tones
    "Build quality feels decent and the camera is okay for the price.",
    "The service was acceptable, nothing special.",
    # Edge cases
    "The weather is nice today.",  # no product aspects → graceful output
    "Everything about this phone is perfect.",  # vague, no specific aspect keyword
]


# -------------------------------------------------------------------------
# 7. Main Execution Function
# -------------------------------------------------------------------------
def main() -> None:
    width = 72
    print("\n" + "-" * width)
    print(f"{_BOLD} ASPECT-BASED SENTIMENT ANALYSIS(ABSA) DEMONSTRATION:{_RESET}")
    print("-" * width)
    analyzer = ABSAAnalyser()
    # -------------------------------------------------------------------------
    # Section i. Detailed per-sentence analysis
    # -------------------------------------------------------------------------
    print(f"\n{_BOLD} -- DETAILED ANALYSIS (with source clauses) -- {_RESET}")
    for sentence in DEMO_SENTENCES:
        result = analyzer.analyze(sentence)
        print_result(result, show_clause=True)
    # -------------------------------------------------------------------------
    # Section ii. Batch Summary
    # -------------------------------------------------------------------------
    results = analyzer.analyze_batch(DEMO_SENTENCES)
    print_summary_table(results)
    # -------------------------------------------------------------------------
    # Section iii. Option al interactive section
    # -------------------------------------------------------------------------
    print(f"\n{_BOLD} -- INTERACTIVE MODE -- {_RESET}")
    print("Type a sentence to analyse it, or press <enter> to Exit.\n")
    while True:
        try:
            user_input = input("Enter text to be analyzed: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            print(" Exiting interactive mode.")
            break
        result = analyzer.analyze(user_input)
        print_result(result, show_clause=True)
    print(f"\n{_BOLD} -- [DONE] : {_RESET} ABSA demo complete.\n")


# ------------------------------------------------------------------------------------------
# 8. Run the script by invoking it's main() function
# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
