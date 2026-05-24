"""
Sample product dataset generator for testing and demonstration
"""

import json
from pathlib import Path


def generate_sample_products():
    """Generate sample e-commerce product data"""

    products = [
        {
            "asin": "PHONE001",
            "title": "Samsung Galaxy S21",
            "brand": "Samsung",
            "price": 799.99,
            "category": "Electronics > Smartphones",
            "features": "6.2-inch Dynamic AMOLED display, 120Hz refresh rate, Snapdragon 888, 5G, 64MP telephoto lens",
            "description": "The Samsung Galaxy S21 features a stunning 6.2-inch Dynamic AMOLED display with 120Hz refresh rate. It comes with a powerful Snapdragon 888 processor, 5G connectivity, and a professional-grade camera system with 12MP wide, 64MP telephoto, and 12MP ultrawide lenses.",
            "combined_description": "Samsung Galaxy S21 | Brand: Samsung | Price: $799.99 | Features: 6.2-inch Dynamic AMOLED display, 120Hz refresh rate, Snapdragon 888, 5G, 64MP telephoto lens | Description: The Samsung Galaxy S21 features a stunning 6.2-inch Dynamic AMOLED display with 120Hz refresh rate. It comes with a powerful Snapdragon 888 processor, 5G connectivity, and a professional-grade camera system."
        },
        {
            "asin": "PHONE002",
            "title": "iPhone 14 Pro",
            "brand": "Apple",
            "price": 999.99,
            "category": "Electronics > Smartphones",
            "features": "6.1-inch Super Retina XDR display, A16 Bionic chip, Pro camera system, ProMotion 120Hz",
            "description": "iPhone 14 Pro features a stunning 6.1-inch Super Retina XDR display. Powered by the A16 Bionic chip, it delivers blazing-fast performance. The Pro camera system includes a 48MP main camera with advanced computational photography.",
            "combined_description": "iPhone 14 Pro | Brand: Apple | Price: $999.99 | Features: 6.1-inch Super Retina XDR display, A16 Bionic chip, Pro camera system, ProMotion 120Hz | Description: iPhone 14 Pro features a stunning 6.1-inch Super Retina XDR display."
        },
        {
            "asin": "WATCH001",
            "title": "Fitbit Charge 5",
            "brand": "Fitbit",
            "price": 179.95,
            "category": "Electronics > Wearables > Fitness Trackers",
            "features": "Built-in GPS, SpO2 tracking, Heart rate monitoring, 7-day battery, Water resistant",
            "description": "Fitbit Charge 5 is a feature-rich fitness tracker with built-in GPS, SpO2 tracking, and comprehensive health metrics. It monitors your heart rate, tracks steps, calories burned, and sleep patterns with up to 7 days of battery life.",
            "combined_description": "Fitbit Charge 5 | Brand: Fitbit | Price: $179.95 | Features: Built-in GPS, SpO2 tracking, Heart rate monitoring, 7-day battery | Description: Fitbit Charge 5 is a feature-rich fitness tracker with built-in GPS and SpO2 tracking."
        },
        {
            "asin": "MIXER001",
            "title": "KitchenAid Artisan Stand Mixer",
            "brand": "KitchenAid",
            "price": 329.99,
            "category": "Home & Kitchen > Kitchen Appliances > Stand Mixers",
            "features": "5-quart stainless steel bowl, 10 speeds, Includes paddle, dough hook, wire whip, NSF certified",
            "description": "KitchenAid Artisan Stand Mixer features a 5-quart stainless steel bowl with 10 speed settings. It comes with flat mixing paddle, dough hook, and wire whip attachments, perfect for baking and cooking tasks. NSF certified for commercial use.",
            "combined_description": "KitchenAid Artisan Stand Mixer | Brand: KitchenAid | Price: $329.99 | Features: 5-quart stainless steel bowl, 10 speeds, NSF certified | Description: KitchenAid Artisan Stand Mixer features a 5-quart stainless steel bowl."
        },
        {
            "asin": "EARBUDS001",
            "title": "Sony WF-1000XM4",
            "brand": "Sony",
            "price": 249.99,
            "category": "Electronics > Audio > Headphones & Earbuds",
            "features": "Active Noise Cancellation, 8-hour battery, LDAC support, Multipoint connection, IPX4 water resistant",
            "description": "Sony WF-1000XM4 truly wireless earbuds with industry-leading active noise cancellation. Features 8-hour battery life, LDAC support for high-quality audio, and seamless multipoint connection between devices.",
            "combined_description": "Sony WF-1000XM4 | Brand: Sony | Price: $249.99 | Features: Active Noise Cancellation, 8-hour battery, LDAC support | Description: Sony WF-1000XM4 truly wireless earbuds with industry-leading active noise cancellation."
        },
        {
            "asin": "TABLET001",
            "title": "iPad Pro 12.9-inch",
            "brand": "Apple",
            "price": 1099.99,
            "category": "Electronics > Tablets",
            "features": "12.9-inch Liquid Retina display, M2 chip, 128GB storage, 5G support, ProMotion 120Hz",
            "description": "iPad Pro 12.9-inch with stunning Liquid Retina display and powerful M2 chip. Perfect for creative professionals with 5G support, ProMotion 120Hz, and support for Apple Pencil and Magic Keyboard.",
            "combined_description": "iPad Pro 12.9-inch | Brand: Apple | Price: $1099.99 | Features: M2 chip, Liquid Retina display, 5G support, ProMotion 120Hz | Description: iPad Pro 12.9-inch with stunning display and powerful M2 chip."
        },
        {
            "asin": "CAMERA001",
            "title": "Canon EOS R5",
            "brand": "Canon",
            "price": 3499.99,
            "category": "Electronics > Cameras > Digital Cameras",
            "features": "45MP full-frame sensor, 4K 120fps video, Canon EF mount, Weather-sealed body, In-body stabilization",
            "description": "Canon EOS R5 is a professional mirrorless camera with 45MP full-frame sensor capable of 4K 120fps video recording. Features weather-sealed body, in-body image stabilization, and advanced autofocus system.",
            "combined_description": "Canon EOS R5 | Brand: Canon | Price: $3499.99 | Features: 45MP sensor, 4K 120fps, Weather-sealed, In-body stabilization | Description: Canon EOS R5 is a professional mirrorless camera."
        },
        {
            "asin": "SPEAKER001",
            "title": "Sonos Move",
            "brand": "Sonos",
            "price": 399.99,
            "category": "Electronics > Audio > Speakers",
            "features": "Portable, WiFi and Bluetooth, 10-hour battery, Weather-resistant, Stereo pairing support",
            "description": "Sonos Move is a portable smart speaker with WiFi and Bluetooth connectivity. Features 10-hour battery life, weather-resistant design, and can be used both indoors and outdoors with exceptional sound quality.",
            "combined_description": "Sonos Move | Brand: Sonos | Price: $399.99 | Features: Portable, WiFi & Bluetooth, 10-hour battery, Weather-resistant | Description: Sonos Move is a portable smart speaker with excellent sound."
        },
        {
            "asin": "LAPTOP001",
            "title": "MacBook Pro 16-inch",
            "brand": "Apple",
            "price": 2499.99,
            "category": "Electronics > Computers > Laptops",
            "features": "M2 Max chip, 16-inch Liquid Retina XDR display, 512GB SSD, 16GB RAM, 17-hour battery",
            "description": "MacBook Pro 16-inch with powerful M2 Max chip, stunning Liquid Retina XDR display, and impressive performance for professionals. Features up to 17-hour battery life and advanced thermal design.",
            "combined_description": "MacBook Pro 16-inch | Brand: Apple | Price: $2499.99 | Features: M2 Max, Liquid Retina XDR, 512GB SSD, 17-hour battery | Description: MacBook Pro 16-inch with powerful M2 Max chip."
        },
        {
            "asin": "DRONE001",
            "title": "DJI Air 2S",
            "brand": "DJI",
            "price": 999.99,
            "category": "Electronics > Camera Drones",
            "features": "1-inch CMOS sensor, 5.1K video, 31-minute flight time, 5G connectivity, Intelligent Flight Modes",
            "description": "DJI Air 2S is a compact drone with 1-inch CMOS sensor capable of 5.1K video recording. Features 31-minute flight time, advanced obstacle avoidance, and intelligent flight modes for content creators.",
            "combined_description": "DJI Air 2S | Brand: DJI | Price: $999.99 | Features: 1-inch CMOS sensor, 5.1K video, 31-minute flight | Description: DJI Air 2S is a compact drone for content creators."
        }
    ]

    return products


def save_sample_products(output_path="data/raw/sample_products.json"):
    """
    Save sample products to JSON file

    Args:
        output_path: Path to save the products
    """
    products = generate_sample_products()

    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save as JSONL (one product per line)
    with open(output_path, 'w') as f:
        for product in products:
            f.write(json.dumps(product) + '\n')

    print(f"Saved {len(products)} sample products to {output_path}")
    return products


if __name__ == "__main__":
    save_sample_products()
