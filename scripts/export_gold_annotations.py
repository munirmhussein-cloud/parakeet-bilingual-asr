from src.annotation.gold_export import export_gold_annotations
from src.annotation.review_store import ReviewStore


if __name__ == "__main__":
    records = ReviewStore().load()
    document = export_gold_annotations(records)
    print(f"Exported {document['record_count']} gold annotations.")
