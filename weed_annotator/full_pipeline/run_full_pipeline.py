import json
import os
import shutil

import logging
logging.basicConfig(format='%(asctime)s %(message)s')

from weed_annotator.semantic_segmentation import utils
from weed_annotator.semantic_segmentation.train import train
from weed_annotator.semantic_segmentation.inference import inference
from weed_annotator.post_processing.post_process_masks import post_process_masks
from weed_annotator.full_pipeline.mask_proposal_evaluator import MaskProposalsEvaluator
from weed_annotator.image_composition.compose_imgs import compose_images

if __name__ == "__main__":
    # create logger
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)

    # Setting seed for reproducability
    utils.set_seeds()

    pipeline_config = json.load(open("configs/weed_annotator.json"))

    # Image Composition
    if pipeline_config["image_composition"]["enable"]:
        logger.info("Generating image compositions for training.")
        img_comp_config = json.load(open("configs/image_composition.json"))
        compose_images(img_comp_config)
        train_folder = img_comp_config["folders"]["out_folder"]
    else:
        train_folder = pipeline_config["image_composition"]["reuse"]

    # Training Semantic Segmemntation
    train_config = json.load(open("configs/seg_config.json"))
    if pipeline_config["sem_segmentation"]["enable_train"]:
        train_config["data"]["train_data"] = [train_folder]
        logger.info(f"Training semantic segmentation model on: {train_folder}.")
        train(train_config)
        log_folder = f"{train_config['logging_path']}/{train_config['train_ident']}"
    else:
        log_folder = pipeline_config["sem_segmentation"]["reuse_model"]

    # Inference
    input_data = pipeline_config["input_imgs"]
    if pipeline_config["sem_segmentation"]["enable_inference"]:
        logger.info(f"Generating mask predictions for: {input_data}.")
        mp_raw = "/tmp/mask_proposals/raw"
        os.makedirs(mp_raw)
        inference(f"{log_folder}/config.json", f"{log_folder}/checkpoints/last.pth", input_data, mp_raw)
    else:
        mp_raw = pipeline_config["sem_segmentation"]["reuse_masks"]

    # Postprocess
    if pipeline_config["post_processing"]["enable"]:
        logger.info("Post-processing mask predictions.")
        mp_pp = pipeline_config["mask_proposals"]
        os.makedirs(mp_pp, exist_ok=True)
        post_process_masks(f"{input_data}", mp_raw, mp_pp)
    else:
        mp_pp = pipeline_config["post_processing"]["reuse"]

    # Evaluation
    if pipeline_config["enable_evaluation"] and os.path.exists(f"{input_data}/annotations.xml"):
        logger.info(f"Evaluation of pipeline performance on: {input_data}.")
        me = MaskProposalsEvaluator(input_data, train_config["data"]["weed_label"])
        result_raw = me.evaluate(mp_raw)
        with open(f"{log_folder}/eval_raw.json", 'w') as f:
            json.dump(result_raw, f)
        result_pp = me.evaluate(mp_pp)
        with open(f"{log_folder}/eval_pp.json", 'w') as f:
            json.dump(result_pp, f)

    # Cleanup
    if pipeline_config["sem_segmentation"]["enable_inference"]:
        shutil.rmtree(f"{mp_raw}")






