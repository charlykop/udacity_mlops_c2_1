import mlflow
import os
import hydra
from omegaconf import DictConfig


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # _ nutzt man, da man nicht interessiert ist an den Output (nach der Konvention)
    _ = mlflow.run(
        # Vorteil os.path.join ist, dass es Plattform (Windosw, IOS, Linux) unabhaengig ist
        os.path.join(root_path, "download_data"),
        # Entry point ist main (wird in MLproject file von download_data definiert)
        "main",
        parameters={
            # ruft Information aus config.yaml (Hydra Funktionalität) von Pipeline ab
            # ["data"]["file_url"] beschreibt die Stelle in der config.yaml, wo die Informationen stehen
            "file_url": config["data"]["file_url"],
            "artifact_name": "iris.csv",
            "artifact_type": "raw_data",
            "artifact_description": "Input data"
        },
    )

    ##################
    # Your code here: use the artifact we created in the previous step as input for the `process_data` step
    # and produce a new artifact called "cleaned_data".
    # NOTE: use os.path.join(root_path, "process_data") to get the path
    # to the "process_data" component
    ##################

    _ = mlflow.run(
        os.path.join(root_path, "process_data"),
        "main",
        # Parameter koennen auch in MLproject Datei von process_data nachgeschaut werden
        parameters={
            # Achtung: Version muss festgelegt werden
            "input_artifact": "iris.csv:latest",
            "artifact_name": "cleaned_data.csv",
            "artifact_type": "processed_data",
            "artifact_description": "Cleaned data"
        },
    )



if __name__ == "__main__":
    go()
