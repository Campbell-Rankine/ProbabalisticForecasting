from gluonts.time_feature import (
    time_features_from_frequency_str,
    TimeFeature,
    get_lags_for_frequency,
)
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
    ConcatFeatures,
    RenameFields,
)

from transformers import PretrainedConfig
from gluonts.transform.sampler import InstanceSampler
from typing import Optional
import numpy as np


def create_data_transformation(
    freq: str,
    config: PretrainedConfig,
    remove_field_names: Optional[list] = [],
    dynamic_real_names: Optional[list] = [],
    static_real_name: Optional[str] = None,
    static_categorical_names: Optional[list] = [],
) -> Transformation:
    # create data transformation
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)

    chain = Chain(
        # Step 1: remove fields
        [RemoveFields(field_names=remove_field_names)]
        +
        # Step 2: add the TS data
        (
            [AsNumpyArray(field="open", expected_ndim=1, dtype=np.float32)]
            + [AsNumpyArray(field="high", expected_ndim=1, dtype=np.float32)]
            + [AsNumpyArray(field="low", expected_ndim=1, dtype=np.float32)]
            + [AsNumpyArray(field="volume", expected_ndim=1, dtype=np.float32)]
            + [
                AsNumpyArray(
                    field=FieldName.TARGET,
                    # we expect an extra dim for the multivariate case:
                    expected_ndim=1,
                    dtype=np.float32,
                )
            ]
            + [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    # we expect an extra dim for the multivariate case:
                    expected_ndim=1,
                    dtype=int,
                )
            ]
        )
        # Step 4: initialize target
        + [
            # Step 5: Fill Nan, create observed value
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # Step 6: Add temporal features based on the frequency string
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(freq),
                pred_length=config.prediction_length,
            ),
            # Step 7: Add Age feature
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
            # Step 8: Vstack temporal features
            VstackFeatures(
                output_field=FieldName.FEAT_DYNAMIC_REAL,
                input_fields=[
                    "open",
                    "high",
                    "low",
                    "volume",
                ],
            ),
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[
                    FieldName.FEAT_TIME,
                    FieldName.FEAT_AGE,
                ],
            ),
            # Step 9: rename fields for HFTransformer
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask",
                }
            ),
        ]
    )
    return chain


from gluonts.transform.sampler import InstanceSampler
from typing import Optional


def create_instance_splitter(
    config: PretrainedConfig,
    mode: str,
    train_sampler: Optional[InstanceSampler] = None,
    validation_sampler: Optional[InstanceSampler] = None,
) -> Transformation:
    assert mode in ["train", "validation", "test"]

    instance_sampler = {
        "train": train_sampler
        or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=config.prediction_length
        ),
        "validation": validation_sampler
        or ValidationSplitSampler(min_future=config.prediction_length),
        "test": TestSplitSampler(),
    }[mode]

    return InstanceSplitter(
        target_field="values",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config.context_length + max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=["time_features", "observed_mask"],
    )
