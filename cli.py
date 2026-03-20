#!/usr/bin/env python3
"""
cli.py — command-line interface for GeoIdentifier

Usage:
    python cli.py photo.jpg
    python cli.py photo.jpg --top-k 3 --verbose
"""

import argparse, logging, sys
from src.identifier import GeoIdentifier

def main():
    parser = argparse.ArgumentParser(description="Identify location from an image.")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")

    identifier = GeoIdentifier()
    print(f"\n🔍 Identifying: {args.image}\n")
    result = identifier.identify(args.image, top_k=args.top_k)

    if not result.predictions:
        print("❌ Could not determine location.")
        sys.exit(1)

    print(f"Strategy: {result.strategy_used}\n{'=' * 60}")
    for i, pred in enumerate(result.predictions[:args.top_k]):
        icon = "📍" if pred.source == "exif" else "🧠"
        print(f"{icon} #{i+1}  [{pred.source.upper()}]  confidence={pred.confidence:.3f}")
        print(f"   Address : {pred.address}")
        print(f"   Coords  : {pred.lat:.5f}, {pred.lon:.5f}")
        print(f"   Maps    : https://maps.google.com/?q={pred.lat},{pred.lon}\n")

if __name__ == "__main__":
    main()
