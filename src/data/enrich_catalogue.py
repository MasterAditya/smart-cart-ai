
"""Enrich synthetic product metadata without changing recommendation data."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"

# Each category has a pool of realistic product names.
PRODUCT_NAMES = {
    "Electronics": [
        "Wireless Bluetooth Earbuds",
        "Portable Bluetooth Speaker",
        "Smart Fitness Watch",
        "Noise Cancelling Headphones",
        "Compact Power Bank",
        "Mechanical Gaming Keyboard",
        "Wireless Optical Mouse",
        "HD Webcam",
        "Smart LED Desk Lamp",
        "Portable SSD Drive",
    ],
    "Clothing": [
        "Classic Oxford Shirt",
        "Relaxed Fit Cotton Hoodie",
        "Everyday Casual T-Shirt",
        "Slim Fit Denim Jeans",
        "Lightweight Summer Jacket",
        "Formal Office Trousers",
        "Cotton Polo Shirt",
        "Comfort Fit Joggers",
        "Textured Knit Sweater",
        "Everyday Linen Shirt",
    ],
    "Home": [
        "Ceramic Table Lamp",
        "Modular Storage Organizer",
        "Minimalist Wall Clock",
        "Cotton Cushion Cover Set",
        "Modern Kitchen Container Set",
        "Wooden Serving Tray",
        "Compact Desk Organizer",
        "Decorative Table Vase",
        "Soft Cotton Bedsheet Set",
        "Contemporary Wall Shelf",
    ],
    "Books": [
        "Foundations of Modern Science",
        "The Art of Deep Learning",
        "A Beginner's Guide to History",
        "The Mystery of the Lost Archive",
        "Learning Python Step by Step",
        "The Psychology of Everyday Life",
        "Introduction to Data Science",
        "The Complete Guide to Astronomy",
        "Stories from Around the World",
        "Principles of Creative Thinking",
    ],
    "Sports": [
        "FlexRun Training Shoes",
        "Adjustable Resistance Bands",
        "Professional Yoga Mat",
        "Lightweight Running Shorts",
        "Stainless Steel Sports Bottle",
        "Compact Fitness Dumbbells",
        "Outdoor Cycling Gloves",
        "Performance Training T-Shirt",
        "Exercise Jump Rope",
        "Sports Gym Duffel Bag",
    ],
    "Beauty": [
        "Hydrating Face Serum",
        "Daily Repair Shampoo",
        "Gentle Foaming Face Wash",
        "Nourishing Body Lotion",
        "Vitamin C Brightening Cream",
        "Moisture Repair Hair Mask",
        "Daily Sunscreen Lotion",
        "Refreshing Facial Toner",
        "Overnight Skin Repair Cream",
        "Natural Lip Care Balm",
    ],
    "Toys": [
        "Creative Building Blocks",
        "Junior Science Discovery Kit",
        "Wooden Puzzle Adventure",
        "Creative Art and Craft Set",
        "Interactive Learning Board",
        "Colorful Stacking Rings",
        "Mini Construction Vehicle Set",
        "Educational Number Puzzle",
        "Magnetic Building Tiles",
        "Animal Discovery Play Set",
    ],
    "Accessories": [
        "Classic Leather Sling Bag",
        "Minimalist Travel Wallet",
        "Everyday Canvas Backpack",
        "Stainless Steel Wristwatch",
        "Compact Card Holder",
        "Premium Laptop Sleeve",
        "Classic Aviator Sunglasses",
        "Adjustable Leather Belt",
        "Travel Organizer Pouch",
        "Minimalist Keychain Set",
    ],
}

BRANDS = {
    "Electronics": ["NovaTech", "Soundora", "VoltEdge", "PixelWorks", "Auralink"],
    "Clothing": ["ThreadTheory", "UrbanWeave", "Northline", "CottonLane", "ModeCraft"],
    "Home": ["CasaLiving", "OakAndStone", "HavenWorks", "NordNest", "HomeTheory"],
    "Books": ["PageTurner", "BrightLeaf", "KnowledgeHouse", "Inkstone", "OpenChapter"],
    "Sports": ["MotionLab", "FlexFit", "PeakMotion", "ActiveCore", "StrideWorks"],
    "Beauty": ["GlowTheory", "PureBloom", "DermaNest", "SkinKind", "BotaniCare"],
    "Toys": ["PlayOrbit", "WonderWorks", "LittleGenius", "BrightMinds", "FunFoundry"],
    "Accessories": ["UrbanCraft", "CarryCulture", "Forma", "EverydayEdit", "NomadWorks"],
}

DESCRIPTION_TEMPLATES = {
    "Electronics": [
        "Designed for convenient everyday use, this {name_lower} combines practical functionality with a clean, modern design.",
        "A versatile technology essential featuring a compact design and useful features for daily routines.",
        "Bring convenience to your setup with this thoughtfully designed electronic accessory.",
    ],
    "Clothing": [
        "Made for everyday versatility, this {name_lower} offers a comfortable fit and an easy-to-style look.",
        "A wardrobe essential designed to work across casual outings and everyday occasions.",
        "Combining a timeless silhouette with practical comfort, this piece is easy to wear throughout the day.",
    ],
    "Home": [
        "Bring practical style to your space with this thoughtfully designed home essential.",
        "A functional home accessory with a contemporary look, suited to modern everyday living.",
        "Designed to complement your interiors while adding convenience to your daily routine.",
    ],
    "Books": [
        "Explore key ideas and engaging perspectives in this thoughtfully presented book.",
        "A reading selection for curious minds, with concepts and insights to explore at your own pace.",
        "Discover an engaging learning experience through this carefully titled publication.",
    ],
    "Sports": [
        "Support your active routine with practical design suited to training and everyday movement.",
        "Designed for fitness enthusiasts seeking comfort and functionality during regular exercise.",
        "A useful addition to your workout essentials, combining practical features with an active lifestyle design.",
    ],
    "Beauty": [
        "A daily personal-care essential designed to fit conveniently into your self-care routine.",
        "Made for everyday beauty routines, with a practical format for regular personal care.",
        "A thoughtfully presented beauty essential for a simple and convenient daily routine.",
    ],
    "Toys": [
        "Encourage curiosity and creative play with this engaging activity for young learners.",
        "Designed to make playtime enjoyable while encouraging exploration and imagination.",
        "A playful addition to a child's collection, combining discovery and hands-on activity.",
    ],
    "Accessories": [
        "Complete your everyday style with this versatile accessory, designed with practicality in mind.",
        "A useful everyday companion combining a clean design with convenient functionality.",
        "Designed to complement daily outfits and routines with a simple, versatile look.",
    ],
}


def enrich_catalogue():

    products = pd.read_csv(RAW_DIR / "products.csv")

    required = {
        "product_id", "category", "price", "average_rating"
    }

    missing = required - set(products.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not products["product_id"].is_unique:
        raise ValueError("Product IDs must be unique.")

    enriched = products.copy()

    # Deterministic naming: same product ID always gets the same metadata.
    category_positions = (
        enriched.groupby("category").cumcount()
    )

    new_names = []
    new_brands = []
    new_descriptions = []

    for position, (_, product) in zip(
        category_positions, enriched.iterrows()
    ):

        category = product["category"]

        if category not in PRODUCT_NAMES:
            raise ValueError(f"Unknown category: {category}")

        names = PRODUCT_NAMES[category]
        brands = BRANDS[category]
        descriptions = DESCRIPTION_TEMPLATES[category]

        name = names[position % len(names)]

        # Add a distinctive series name when the pool repeats.
        cycle = position // len(names)
        if cycle > 0:
            name = f"{name} - Series {cycle + 1}"

        brand = brands[position % len(brands)]

        description = descriptions[position % len(descriptions)]

        description = description.format(
            name_lower=name.lower()
        )

        new_names.append(name)
        new_brands.append(brand)
        new_descriptions.append(description)

    enriched["product_name"] = new_names
    enriched["brand"] = new_brands
    enriched["description"] = new_descriptions

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "products_enriched.csv"
    enriched.to_csv(output_path, index=False)

    print("Catalogue enrichment completed.")
    print(f"Products enriched: {len(enriched)}")
    print(f"Unique product IDs: {enriched['product_id'].nunique()}")
    print(f"Output: {output_path}")

    print("\nSample enriched products:")
    print(
        enriched[
            ["product_id", "product_name", "brand", "description"]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    enrich_catalogue()
