import json
import argparse


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def numeric_bidder_key(k):
    try:
        return int(str(k).split('_')[-1])
    except Exception:
        return 0


def compare(cca_parsed_path, mlcca_parsed_path):
    cca = load_json(cca_parsed_path)
    mlcca = load_json(mlcca_parsed_path)

    bidders = sorted(set(list(cca.keys()) + list(mlcca.keys())), key=numeric_bidder_key)
    combined = {}
    for b in bidders:
        cca_items = cca.get(b, {}).get("items", [])
        ml_items = mlcca.get(b, {}).get("items", [])
        combined[b] = {
            "cca": cca_items,
            "mlcca": ml_items,
            "same": sorted(cca_items) == sorted(ml_items)
        }
    return combined


def main():
    parser = argparse.ArgumentParser(description="Compare CCA vs MLCCA allocations for a round.")
    parser.add_argument("--cca_parsed", required=True, help="Path to parsed CCA JSON (from parse_allocations.py)")
    parser.add_argument("--mlcca_parsed", required=True, help="Path to parsed MLCCA JSON (from parse_allocations.py)")
    parser.add_argument("--output", required=True, help="Path to save comparison JSON")
    args = parser.parse_args()

    combined = compare(args.cca_parsed, args.mlcca_parsed)
    with open(args.output, 'w') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"Saved comparison to {args.output}")


if __name__ == "__main__":
    main()