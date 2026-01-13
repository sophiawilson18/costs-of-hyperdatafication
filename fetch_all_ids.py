from huggingface_hub import list_datasets

# Open file once and stream dataset IDs into it
with open("all_dataset_ids.txt", "w") as f:
    for ds in list_datasets():   # this yields datasets lazily
        f.write(ds.id + "\n")

print("Finished writing all dataset IDs to all_dataset_ids.txt")