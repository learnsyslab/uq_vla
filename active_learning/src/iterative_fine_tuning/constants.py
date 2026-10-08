"""Shared constants for iterative fine-tuning."""

ITERATIVE_CONFIG_NAME = "iterative_fine_tuning_config.json"
RUN_STATE_NAME = "run_state.json"
INITIAL_EVALUATION_DIRNAME = "initial_evaluation"
INITIAL_EVALUATION_NAME = "initial_evaluation.json"
SELECTION_MANIFEST_NAME = "selection_manifest.json"
TRAINING_MANIFEST_NAME = "training_manifest.json"
CANDIDATE_SCORES_NAME = "candidate_scores.json"

ROUND_DIR_TEMPLATE = "round_{round_index:03d}"
TRAINING_DIRNAME = "training"
MEMBER_TRAINING_REQUEST_NAME = "member_training_request.json"
MEMBER_TRAINING_RESULT_NAME = "member_training_result.json"
MEMBER_TRAINING_LOG_NAME = "member_training.log"
MEMBER_EVALUATION_NAME = "member_evaluation.json"
MEMBER_EVALUATION_REQUEST_NAME = "member_evaluation_request.json"
MEMBER_EVALUATION_LOG_NAME = "member_evaluation.log"
ROUND_EVALUATION_NAME = "round_evaluation.json"
RUN_EVALUATION_NAME = "run_evaluation.json"

STAGE_SELECT = "select"
STAGE_TRAIN = "train"
VALID_STAGES = (STAGE_SELECT, STAGE_TRAIN)

POOL_HISTORY = "history"
POOL_NEW = "new"
POOL_ALL = "all"
VALID_POOL_NAMES = (POOL_HISTORY, POOL_NEW, POOL_ALL)
