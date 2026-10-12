"""Run the full pipeline: generate -> detect -> evaluate."""
from pathlib import Path
from src import detect, evaluate, generate
ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"
START = "<!-- RESULTS_TABLE_START -->"
END = "<!-- RESULTS_TABLE_END -->"
def update_readme(table):
    """Put the comparison table between the markers in README.md."""
    if not README.exists():
        print("README.md not found yet - skipping README update.")
        return
    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        print("README markers not found - skipping README update.")
        return
    before = text.split(START)[0]
    after = text.split(END)[1]
    new_text = before + START + "\n" + evaluate.markdown_table(table) + "\n" + END + after
    README.write_text(new_text, encoding="utf-8")
    print("README.md results table updated.")
def main():
    print("=== Step 1/3: Generating data ===")
    generate.main()
    print("\n=== Step 2/3: Features + detection ===")
    detect.main()
    print("\n=== Step 3/3: Evaluation ===")
    table = evaluate.main()
    update_readme(table)
    print("\nAll done.")
if __name__ == "__main__":
    main()
