import sys
import csv

def parse_mnova_line(filepath):
    """Parse a MestreNova peak export (single tab-separated line)."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read().strip()

    tokens = text.split("\t")
    if len(tokens) < 4:
        raise ValueError("File format not recognized: expected tab-separated (ppm, intensity) pairs")

    # Skip the first two fields (index + filename)
    data = tokens[2:]

    # Parse alternating ppm, intensity pairs
    peaks = []
    for i in range(0, len(data), 2):
        try:
            ppm = float(data[i].replace(",", "."))
            intensity = float(data[i + 1].replace(",", "."))
            peaks.append((ppm, intensity))
        except (ValueError, IndexError):
            break

    return peaks

def main():
    if len(sys.argv) < 2:
        print("Usage: python mnova_to_peaks.py input.txt")
        sys.exit(1)

    infile = sys.argv[1]
    outfile = infile.rsplit(".", 1)[0] + "_peaks.txt"

    peaks = parse_mnova_line(infile)

    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ppm", "intensity"])
        writer.writerows(peaks)

    print(f"Wrote {len(peaks)} peaks to {outfile}")

if __name__ == "__main__":
    main()
