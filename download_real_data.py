import os
from datasets import load_dataset
from tqdm import tqdm


def main():
    # Folder structure: data/real_data/mixed/images/
    output_dir = "data/real_data"
    category = "mixed"
    subdir = "images"

    save_path = os.path.join(output_dir, category, subdir)
    os.makedirs(save_path, exist_ok=True)

    dataset = load_dataset(
        "frgfm/imagenette",
        "320px",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    limit = 9000
    print(f"Saving {limit} images to {save_path}...")

    count = 0
    for item in tqdm(dataset):
        image = item["image"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        image.save(os.path.join(save_path, f"img_{count:05d}.jpg"))

        count += 1
        if count >= limit:
            break

    print(f"Done! Saved {count} images.")


if __name__ == "__main__":
    main()
