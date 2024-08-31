from glob import glob
import click

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob

@click.command()
@click.option("--dataset-name", "--name", required=True)
def main(dataset_name):
    # to delete a fiftyone dataset using python
    # fo.delete_dataset(dataset_name)

    assert fo.dataset_exists(dataset_name) == False, f"Dataset {dataset_name} already exists"
    images_dir = "dataset/images"

    #################################
    # Create a new fiftyone dataset #
    #################################
    dataset = fo.Dataset.from_images_dir(images_dir, name=dataset_name, persistent=True)

    ############################
    # Compute image embeddings #
    ############################
    # load a pre-trained model from model zoo from fiftyone
    model = foz.load_zoo_model("clip-vit-base32-torch")
    embeddings = dataset.compute_embeddings(model,
                                            embeddings_field="clip_embeddings")

    #############################################
    # Get a 2D visualization  of the emb. space #
    #############################################
    print("[INFO] Computing 2D visualization using embeddings")
    fob.compute_visualization(dataset,
                              embeddings=embeddings,
                              method="umap",  # "umap", "tsne", "pca", etc
                              brain_key=f"clip-embegginds-viz".replace("-","_"))


    # compute similarity
    print("[INFO] Computing similarity using")
    fob.compute_similarity(dataset,
                           embeddings=embeddings,
                           brain_key=f"clip-similarity".replace("-","_"),
                           num_workers=10, progress=True)

    # Compute uniqueness
    print("[INFO] Computing uniqueness ...")
    fob.compute_uniqueness(dataset,
                           uniqueness_field=f"clip-uniqueness".replace("-","_"),
                           embeddings=embeddings,
                           progress=True,
                           batch_size=10,
                           num_workers=10)

    #############################
    # Launch local fiftyone app #
    # Optional                  #
    #############################
    session = fo.launch_app(dataset)
    session.wait(-1)

    # dataset.compute_embeddings(embeddings_field=..., )

if __name__ == "__main__":
    main()