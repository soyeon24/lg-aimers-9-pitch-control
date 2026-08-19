"""v1(기본 47피처) vs v2(파생 70피처)를 두 홀드아웃에서 동시에 비교.

v2는 2024 홀드아웃에서 +55.8이었지만 리더보드에서는 +11.4에 그쳤다.
2023 홀드아웃에서도 같은 방향으로 개선되는지 확인해, 그 +55.8이
실제 신호였는지 한 해짜리 노이즈였는지 가른다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from evaluate import base_frame, dual_holdout
from features import build_features as v2_build


def v1_build(d, priors):
    return base_frame(d)


def v2_build_wrapped(d, priors):
    return v2_build(d, priors)


dual_holdout(v1_build, "v1 기본 47피처")
dual_holdout(v2_build_wrapped, "v2 파생 70피처")
