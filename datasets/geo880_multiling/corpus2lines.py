import os
import re

import ujson


def run():
    languages = ["de", "el", "en", "fa", "id", "sv", "th", "zh"]
    split_path = "geoquery-2012-08-27/splits/split-880/run-0/fold-0/"
    path = "geoquery/"

    # get test ids
    test_id_file = os.path.join(split_path, "test")
    with open(test_id_file, "r") as f:
        test_ids = f.readlines()
        test_ids = [int(x.strip()) for x in test_ids]
        test_ids = set(test_ids)

    print(test_ids)
    testids_outpath = os.path.join(path, "testids.set")
    with open(testids_outpath, "w") as f:
        ujson.dump(test_ids, f)

    # process all languages:
    for lang in languages:
        examples = []
        corpus_path = os.path.join(path, f"geoFunql-{lang}.corpus")
        with open(corpus_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # process lines:
        example = None
        for line in lines:
            if example is None:
                assert(line.startswith("id:"))
            if line.startswith("id:"):
                example_id = int(re.match("id:(.+)$", line).group(1))
                example = {"id": example_id}
            elif line.startswith("nl:"):
                assert(example is not None)
                example["nl"] = line[3:].strip()
            elif line.startswith("mrl:"):
                assert(example is not None)
                example["mrl"] = line[4:].strip()
            elif line.strip() == "":
                example["split"] = "test" if example["id"] in test_ids else "train"
                examples.append(example)
                example = None
            else:
                pass
        with open(os.path.join(path, f"geo-{lang}.json"), "w") as f:
            ujson.dump(examples, f, indent=4)


def try_json():
    path = "geoquery/geo-de.json"
    with open(path, "r") as f:
        data = ujson.load(f)
        for example in data:
            print(example["nl"])
            print(example["mrl"])


if __name__ == '__main__':
    run()
    try_json()