# TODO: Make this better and not so manual to update

from neuralsignal.core.modules.feature_sets.feature_set_zones\
    import FeatureSetZones
from neuralsignal.core.modules.feature_sets.feature_set_base\
    import FeatureSetBase

fs = {
    "zones": FeatureSetZones
}


def make_feature_set(
        feature_set_name: str, cfg: dict
        ) -> FeatureSetBase:

    if feature_set_name in fs:
        return fs[feature_set_name](cfg)
    else:
        raise ValueError(
            f"Feature set {feature_set_name} does not exist"
            )
