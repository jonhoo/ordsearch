import json
import csv

TYPES = ['u8', 'u16', 'u32', 'u64']
SIZES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4069, 8192, 16384, 32768, 65536]
COLLECTIONS = ['ordsearch', 'sorted_vec', 'btreeset']


def main():
    for collection in COLLECTIONS:
        data = read_collection_stat(collection)
        with open(f"target/{collection}.csv", "w") as f:
            writer = csv.DictWriter(f, data[0].keys())
            writer.writeheader()
            writer.writerows(data)


def read_collection_stat(collection: str) -> list[dict[str, int]]:
    result = []
    for size in SIZES:
        row = {'size': size}
        for type_name in TYPES:
            file_name = f"target/criterion/Search {type_name}/{collection}/{size}/new/estimates.json"
            with open(file_name) as f:
                j = json.loads(f.read())
                row[type_name] = j['mean']['point_estimate']
        result.append(row)
    return result


if __name__ == "__main__":
    main()
