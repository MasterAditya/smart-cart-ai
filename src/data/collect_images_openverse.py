
from pathlib import Path
import time
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "processed" / "products_enriched.csv"
OUTPUT_DIR = ROOT / "assets" / "openverse_test"
MANIFEST = ROOT / "reports" / "openverse_test.csv"

API_URL = "https://api.openverse.org/v1/images/"
TEST_LIMIT = 5

def main():
    df = pd.read_csv(CSV_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for index, product in df.head(TEST_LIMIT).iterrows():
        product_id = product["product_id"]
        name = product["product_name"]
        category = product["category"]

        query = f"{name} {category}"
        print(f"\n[{index + 1}/{TEST_LIMIT}] Searching: {query}")

        record = {
            "product_id": product_id,
            "product_name": name,
            "search_query": query,
            "image_path": "",
            "creator": "",
            "source": "",
            "license": "",
            "status": "not_found",
        }

        try:
            response = requests.get(
                API_URL,
                params={
                    "q": query,
                    "page_size": 10,
                    "mature": "false",
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            for image in data.get("results", []):
                image_url = image.get("url")
                license_name = image.get("license")
                creator = image.get("creator")
                source = image.get("foreign_landing_url")

                if not image_url or not license_name:
                    continue

                image_response = requests.get(
                    image_url,
                    timeout=30,
                    headers={"User-Agent": "SmartCartAI/1.0"},
                )
                image_response.raise_for_status()

                content_type = image_response.headers.get(
                    "Content-Type", ""
                ).lower()

                if not content_type.startswith("image/"):
                    continue

                extension = {
                    "image/jpeg": ".jpg",
                    "image/png": ".png",
                    "image/webp": ".webp",
                }.get(content_type.split(";")[0], ".img")

                product_dir = OUTPUT_DIR / product_id
                product_dir.mkdir(parents=True, exist_ok=True)

                image_path = product_dir / f"{product_id}{extension}"
                image_path.write_bytes(image_response.content)

                record.update({
                    "image_path": str(image_path.relative_to(ROOT)),
                    "creator": creator or "",
                    "source": source or "",
                    "license": license_name,
                    "status": "downloaded",
                })

                print("Downloaded:", image_path)
                break

            if record["status"] != "downloaded":
                print("No usable image found.")

        except Exception as error:
            record["status"] = "error"
            print("Error:", error)

        results.append(record)
        pd.DataFrame(results).to_csv(MANIFEST, index=False)
        time.sleep(1)

    print("\nTest finished.")
    print("Manifest:", MANIFEST)
    print("Successful downloads:",
          sum(r["status"] == "downloaded" for r in results),
          "/", TEST_LIMIT)

if __name__ == "__main__":
    main()
