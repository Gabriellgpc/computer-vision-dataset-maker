import click
import fiftyone.brain as fob
import fiftyone as fo
from ov_florence2_helper import OVFlorence2Model
from transformers import AutoProcessor
from PIL import Image

import numpy as np

def normalize_bbox(bbox, image_height, image_width):
    x1, y1, x2, y2 = bbox
    return (x1 / image_width, y1 / image_height,
            (x2 - x1) / image_width, (y2 - y1) / image_height)

def run_inference(sample_collection, model_path):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = OVFlorence2Model(model_path, "AUTO")

    for sample in sample_collection.iter_samples(autosave=True, progress=True):
        try:
            # Load image
            image = Image.open(sample.filepath)
            width, height = image.width, image.height

            # Extract image-features (embedding)
            inputs = processor(text="<OD>", images=image, return_tensors="pt")
            image_features = model.encode_image(inputs["pixel_values"])

            # Object detection and caption inference in a single loop
            detections, caption = [], None
            for task in ["<OD>", "<CAPTION>"]:
                if task == "<CAPTION>":
                    inputs = processor(text=task, images=image, return_tensors="pt")
                generated_ids = model.generate(input_ids=inputs["input_ids"],
                                            pixel_values=inputs["pixel_values"],
                                            max_new_tokens=1024,
                                            do_sample=False,
                                            image_features=image_features,
                                            num_beams=3)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed_answer  = processor.post_process_generation(generated_text, task=task, image_size=(width, height))

                if task == "<OD>":
                    for idx, bbox in enumerate(parsed_answer[task]['bboxes']):
                        label = parsed_answer[task]["labels"][idx]
                        normalized_bbox = normalize_bbox(bbox, height, width)
                        detections.append(fo.Detection(label=label, bounding_box=normalized_bbox))
                else:
                    caption = parsed_answer[task]

            # Add predictions to sample
            sample["detections"] = fo.Detections(detections=detections)
            sample["caption"] = caption
            sample["florence2_image_feats"] = image_features.reshape(-1) # flatting image features
        except Exception as e:
            continue

@click.command()
@click.option("--dataset-name",
              "--name",
              required=True,
              prompt="Name of the dataset?")
@click.option("--model-path",
              "-m",
              required=False,
              default="Florence-2-base")
def main(dataset_name, model_path):
    assert fo.dataset_exists(dataset_name), f"Dataset {dataset_name} does not exist yet."
    dataset = fo.load_dataset(dataset_name)
    run_inference(dataset, model_path)

    ###################################################################
    # Get 2D embedding space visualization from florence2-image-feats #
    ###################################################################
    # recovery embeddings (image features) from the sample field "florence2_image_feats", populated during "run_inference"
    florence_embeddings = dataset.values(field_or_expr="florence2_image_feats")
    florence_embeddings = np.array(florence_embeddings).reshape(len(dataset), -1)

    print("[INFO] Computing 2D visualization using embeddings")
    fob.compute_visualization(dataset,
                              embeddings=florence_embeddings,
                              method="umap",
                              brain_key="florence2_embegginds_viz")

if __name__ == '__main__':
    main()
