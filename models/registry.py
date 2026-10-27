"""Active FiLM Transformer baseline and its fixed task registry."""

TASKS = ("static", "temporal", "dynamic", "text_video")
BRANCHES = ("technical", "motion", "text_video")
ATTRIBUTES = ("technical_projector", "motion_projector", "tv_projector")
WIDTHS = (512, 256, 768)
# Historical serialization identifier, paired with a single supported variant.
BASELINE_NAME = "multimodal_readout_variants"
BASELINE_VARIANT = "film_transformer"
