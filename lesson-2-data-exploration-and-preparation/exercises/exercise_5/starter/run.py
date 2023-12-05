#!/usr/bin/env python
import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(project="exercise_5", job_type="process_data")

    logger.info("Getting artifact")
    artifact = run.use_artifact(args.input_artifact)
    local_path = artifact.file()
    logger.info(f"Artifact was downloaded to {local_path}.")

    logger.info("Transform input artifact to pandas dataframe.")
    df = pd.read_parquet(local_path)

    logger.info("Delete duplicates.")
    df = df.drop_duplicates().reset_index(drop=True)

    logger.info("Add new feature - combination of features titel and son_name.")
    df['title'].fillna(value='', inplace=True)
    df['song_name'].fillna(value='', inplace=True)
    df['text_feature'] = df['title'] + ' ' + df['song_name']

    logger.info("Tranform output artifact to csv.")
    filename = args.artifact_name
    df.to_csv(filename)

    logger.info("Creating artifact.")
    output_artifact = wandb.Artifact(
        name=args.artifact_name, 
        type=args.artifact_type, 
        description=args.artifact_description
    ) 

    logger.info("Add artifact and log artifact.")
    output_artifact.add_file(filename)
    run.log_artifact(output_artifact)
    
    # run.finish() --> nicht notwendig, wenn nur ein run im Skript

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)
