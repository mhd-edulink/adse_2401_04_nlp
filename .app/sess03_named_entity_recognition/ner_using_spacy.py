"""
=================================================================================================
Python script to demonstrate NER (named Entity Recognition) using the Spacy library
=================================================================================================
This script identifies and classifies 'named entities' using spacy's medium English model
into predefined categories such as:
    - PERSON    : People's names
    - ORG       : Companies, agencies, institutions
    - GPE       : Geopolitical entities such as (countries, cities, states, provinces & so on)
    - LOC       : Non-GPE locations (mountains, rivers, waterfalls, & so on)
    - DATE      : Absolute or relative dates / periods
    - TIME      : Times smaller than a day
    - MONEY     : Monetary values
    - FACILITY  : Buildings, airports, highways, and so on
    - EVENT     : Buildings, airports, highways, and so on
    - WORLD     : Named events (elections, battles and so on)

This script uses spaCy's pre-trained English model (`en_core_web_md`) to
perform NER on two sample paragraphs, then displays results in multiple ways:
  1. Token-level entity printing
  2. Grouped results by entity type
  3. Rich ASCII table with colours (via the `rich` library if available)
  4. spaCy's built-in displaCy HTML visualiser saved to file

Requirements:
  pip install spacy
  python -m spacy download en_core_web_md

  # Optional — for colour table output:
  pip install rich
"""

# -----------------------------------------------------------------------------------------------
# 0. Import the required modules
# -----------------------------------------------------------------------------------------------
import collections
import sys
import spacy
from spacy.lang.en import English


# -----------------------------------------------------------------------------------------------
# 1. Dependancy checks
# -----------------------------------------------------------------------------------------------
def _check_import(module_name: str, install_hint: str) -> object:
    """
    Attempt to import *module_name* and exit with a helpful message on failure.

    Args:
        module_name  : The Python module to import (e.g. "spacy").
        install_hint : The pip command users should run if the module is absent.

    Returns:
        The imported module object.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        print(f"\n[ERROR] {module_name} is not installed.\nRun: {install_hint}\n")
        sys.exit(1)  # Stop further script execution to avoid missing module / library errors


spacy = _check_import("spacy", "pip install spacy && python -m spacy download en_core_web_md")

# 'rich' module is optional - we degrade gracefully if absent
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box as rich_box

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# -----------------------------------------------------------------------------------------------
# 2. Sample Corpurs (text).
# -----------------------------------------------------------------------------------------------
SAMPLE_TEXT = """
Ada Lovelace visited London in September 1843 to present her notes on the
analytical engine at a meeting organised by the Royal Society. During the
event, she spoke with engineers from IBM, and researchers from Microsoft
about the future of computing and artificial intelligence. After the
conference, she travelled by train to Manchester and stayed at The Midland
Hotel for three nights before returning home.

Last Friday, David Beckham attended a charity fundraiser at Wembley Stadium
alongside representatives from UNICEF. The organisers announced that more
than £2 million had been raised to support schools in Kenya and India.
Following the ceremony, Beckham shared photographs on Instagram and thanked
supporters from across the United Kingdom for their generosity.
"""


# -----------------------------------------------------------------------------------------------
# 3. Load spaCy model
# -----------------------------------------------------------------------------------------------
def load_nlp_model(model_name: str = "en_core_web_md"):
    """
    Load a spaCy language model by name.

    spaCy models bundle a tokeniser, POS tagger, dependency parser, and NER
    pipeline component.  The medium English model (en_core_web_md) is moderate and
    suitable for demonstrations; larger models (en_core_web_lg / trf) are more
    accurate for production use.

    Args:
        model_name : The spaCy model identifier string.

    Returns:
        A spaCy `Language` object (the NLP pipeline).
    """
    try:
        nlp = spacy.load(model_name)
        print(f"[INFO] Loaded {model_name} model")
    except OSError:
        print(
            f"[ERROR]: Could not find spaCy model: {model_name}\n[INFO]: Downloading {model_name} now, please wait...")

        # Download the model programattically
        from spacy.cli import download
        try:
            download("en_core_web_md")

            # Try downloading again after download
            nlp = spacy.load("en_core_web_md")
            print(f"[INFO] Successfully downloaded {model_name}")
            return nlp
        except Exception as e:
            print(f"\n[ERROR]: Failed to download spaCy model\n{e}")
            sys.exit(1)  # Stop further script execution due to failed download of the English medium model


# -----------------------------------------------------------------------------------------------
# 4. Function to run NER
# -----------------------------------------------------------------------------------------------
def run_ner(nlp, text: str):
    """
    Run the spaCy NLP pipeline on *text* and return the processed Doc object.

    The Doc object contains:
      - doc.ents  : A tuple of Span objects, one per detected entity.
        - span.text        : The surface string of the entity.
        - span.label_      : The entity type label (e.g. "PERSON").
        - span.start_char  : Character offset (start) in the original text.
        - span.end_char    : Character offset (end) in the original text.
      - doc.sents : Individual sentences (requires the parser component).

    Args:
        nlp  : A loaded spaCy Language pipeline.
        text : The raw input string to analyse.

    Returns:
        A spaCy Doc object.
    """
    doc = nlp(text)
    return doc


# -----------------------------------------------------------------------------------------------
# 5. Display Helpers
# -----------------------------------------------------------------------------------------------
def print_entities_flat(doc):
    """
    Print every detected entity on its own line in a flat, sequential list.

    Format:
        <ENTITY TEXT>  →  <LABEL>  (<start>–<end>)

    This mirrors the order in which entities appear in the source text,
    making it easy to verify extraction against the original document.

    Args:
        doc : A processed spaCy Doc object.
    """
    print("\n" + "=" * 70)
    print(" ENTITIES (sequential order)")
    print("=" * 70)

    if not doc.ents:
        print("No entities found / detected.")
        return  # Stop further function execution

    for ent in doc.ents:
        description = spacy.explain(ent.label_) or "-"
        print(f"{ent.text:<30} -> {ent.label_:<12}"
              f"[{description}]"
              f"chars {ent.start_char} - {ent.end_char}")


def print_entities_grouped(doc) -> None:
    """
    Group entities by their label and print each group alphabetically.

    Useful for quickly seeing *all* organisations or *all* dates together
    without hunting through a flat list.

    Args:
        doc : A processed spaCy Doc object.
    """
    print("\n" + "=" * 70)
    print(" ENTITIES (grouped by type)")
    print("=" * 70)

    grouped: dict[str, list[str]] = collections.defaultdict(list)
    for ent in doc.ents:
        grouped[ent.label_].append(ent.text)

    for label in sorted(grouped.keys()):
        description = spacy.explain(label) or "-"
        entities = ",".join(sorted(set(grouped[label])))  # Remove duplicates and sort
        print(f"\n{label} ({description})"
              f"{entities}")


def print_entities_rich_table(doc) -> None:
    """
    Render a nicely formatted table using the `rich` library (if installed).

    Columns: #  |  Entity Text  |  Type  |  Description  |  Char Span

    Falls back gracefully to a plain-text message if `rich` is not available.

    Args:
        doc : A processed spaCy Doc object.
    """
    print("\n" + "=" * 70)
    print(" ENTITIES (grouped by type)")
    print("=" * 70)

    if not RICH_AVAILABLE:
        print("[ERROR] Rich library not installed. skipping table view"
              " Install with `pip install rich`")
        return
    console = Console()
    table = Table(
        title="Named Entities Detected",
        box=rich.box.ROUNDED,
        header_style="bold cyan",
        show_lines=True,
    )

    # Actual table
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Entity", style="bold whitte", width=28)
    table.add_column("Type", style="bold yellow", width=12)
    table.add_column("Description", style="green", width=30)
    table.add_column("Char Span", style="dim", width=128)

    # Color Map: entity label -> row style
    label_colours = {
        "PERSON"    : "bright_magenta",
        "ORG"       : "bright_blue",
        "GPE"       : "bright_cyan",
        "LOC"       : "cyan",
        "DATE"      : "bright_green",
        "TIME"      : "green",
        "MONEY"     : "bright_yellow",
        "FAC"       : "orange",
        "EVENT"     : "red",
    }

    for ent in enumerate(doc.ents, start=1):
        description = spacy.explain(ent.label_) or "-"
        colour = label_colours.get([ent.label_, "white"])
        table.add_row(
            f"[{colour}]{ent.text}[/{description}]"
            f"[{colour}]{ent.label_}[/{colour}]",
            description,
            f"{ent.start_char} - {ent.end_char}"
        )

    console.print(table)

# Generate HTML document
def save_displacy_html(doc, output_path: str = "ner_displacy.html") -> None:
    """
    Render the NER annotations as colour-coded HTML using spaCy's displaCy
    visualiser and save the result to *output_path*.

    displaCy generates an HTML snippet where each entity span is highlighted
    with a distinct colour per label and a floating label badge.  Open the
    output file in any browser to inspect the results visually.

    Args:
        doc         : A processed spaCy Doc object.
        output_path : Destination file path for the HTML output.
    """
    from spacy import displacy
    html = displacy.render(doc, style="ent", page=True, minify=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    print(f"\n[INFO] displaCy visualisation saved -> {output_path}"
          f"Open in browser to see colour-coded entity spans")

# -----------------------------------------------------------------------------------------------
# 6. Summary Statistics
# -----------------------------------------------------------------------------------------------
def print_summary(doc) -> None:
    """
    Print high-level statistics about the NER results.
    Includes:
        - Total word / token count
        - Total sentence count
        - Total entity count
        - Per-label entity counts (sorted by frequency, descending)
    Args:
        doc: A processed spaCy Doc object.
    """
    print("\n" + "=" * 70)
    print("Summary Statistics:")
    print("=" * 70)

    tokens = [t for t in doc if not t.is_space]
    sentences = list(doc.sents)
    entities = list(doc.ents)

    print(f"Total tokens: {len(tokens)}"
          f"Entities: {len(entities)}")

    if entities:
        counter = collections.Counter(ent.label_ for ent in entities)
        print("\nBeakdown by label:")
        for label, count in counter.most_common():
            description = spacy.explain(label) or "-"
            bar = "█" * count
            print(f"{label:<12} -> {count:>3} {bar} ({description})")
