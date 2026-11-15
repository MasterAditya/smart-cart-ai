
from pathlib import Path
import pandas as pd
from icrawler.builtin import BingImageCrawler

# Project paths
ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "processed" / "products_enriched.csv"
IMAGE_DIR = ROOT / "assets" / "product_images"
MANIFEST_PATH = ROOT / "reports" / "product_images.csv"

# Start with five products for testing
TEST_LIMIT = 5
IMAGES_PER_PRODUCT = 1

def main():
    df = pd.read_csv(CSV_PATH)

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for index, product in df.head(TEST_LIMIT).iterrows():
        product_id = product["product_id"]
        name = product["product_name"]
        category = product["category"]
        brand = product["brand"]

        query = f"{name} {category}"

        product_folder = IMAGE_DIR / product_id
        product_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n[{index + 1}/{TEST_LIMIT}] Searching: {query}")

        try:
            crawler = BingImageCrawler(
                storage={"root_dir": str(product_folder)}
            )

            crawler.crawl(
                keyword=query,
                max_num=IMAGES_PER_PRODUCT,
                filters=None
            )

            images = [
                p for p in product_folder.iterdir()
                if p.is_file() and p.suffix.lower() in
                {".jpg", ".jpeg", ".png", ".webp"}
            ]

            image_path = str(images[0].relative_to(ROOT)) if images else ""

            results.append({
                "product_id": product_id,
                "product_name": name,
                "image_path": image_path,
                "search_query": query,
                "status": "downloaded" if image_path else "not_found"
            })

            print("Result:", image_path or "No image downloaded")

        except Exception as error:
            print("Error:", error)

            results.append({
                "product_id": product_id,
                "product_name": name,
                "image_path": "",
                "search_query": query,
                "status": "error"
            })

    pd.DataFrame(results).to_csv(MANIFEST_PATH, index=False)

    print("\nTest complete.")
    print("Manifest:", MANIFEST_PATH)

if __name__ == "__main__":
    main()
