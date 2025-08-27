import json
from google.cloud import aiplatform, storage
from config.config import PROJECT_ID, REGION, ARTIFACTS_GCS_PREFIX,XGB_DISPLAY_NAME, LSTM_DISPLAY_NAME, NLP_DISPLAY_NAME,ENDPOINT_DISPLAY_NAME, ENDPOINT_MACHINE_TYPE,SKLEARN_IMAGE_URI, XGBOOST_IMAGE_URI, TENSORFLOW_IMAGE_URI

def _read_best_model_type() -> str:
    client = storage.Client(project=PROJECT_ID)
    bucket_name = ARTIFACTS_GCS_PREFIX.split("gs://",1)[1].split("/",1)[0]
    prefix = ARTIFACTS_GCS_PREFIX.split(bucket_name+"/",1)[1]
    blob = client.bucket(bucket_name).blob(f"{prefix}/metrics/metrics.json")
    data = json.loads(blob.download_as_text())
    best = max(data.items(), key=lambda kv: kv[1].get("f1", 0))[0]
    return best

def _artifact_dir_and_container(model_type: str):
    if model_type == "xgb":
        return f"{ARTIFACTS_GCS_PREFIX}/models/xgb", XGB_DISPLAY_NAME, XGBOOST_IMAGE_URI
    if model_type == "nlp":
        return f"{ARTIFACTS_GCS_PREFIX}/models/nlp", NLP_DISPLAY_NAME, SKLEARN_IMAGE_URI
    if model_type == "lstm":
        return f"{ARTIFACTS_GCS_PREFIX}/models/lstm_savedmodel", LSTM_DISPLAY_NAME, TENSORFLOW_IMAGE_URI
    raise ValueError(f"Unknown model_type: {model_type}")

def register_and_deploy_best():
    aiplatform.init(project=PROJECT_ID, location=REGION)
    best = _read_best_model_type()
    artifact_uri, display_name, image_uri = _artifact_dir_and_container(best)

    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=image_uri,
    )
    print(f"Uploaded model: {model.resource_name}")

    eps = list(aiplatform.Endpoint.list(filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"'))
    endpoint = eps[0] if eps else aiplatform.Endpoint.create(display_name=ENDPOINT_DISPLAY_NAME)
    print(f"Endpoint: {endpoint.resource_name}")

    endpoint.deploy(model=model, machine_type=ENDPOINT_MACHINE_TYPE, traffic_percentage=100, sync=False)
    print("Deployment started")
