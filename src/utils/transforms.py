from typing import Union, Collection, Hashable
import monai
from monai.transforms.utils_pytorch_numpy_unification import clip


class ClipIntensityd(monai.transforms.Lambdad):
    def __init__(
            self,
            keys: Union[Collection[Hashable], Hashable],
            i_min: Union[int, float],
            i_max: Union[int, float]
    ):
        self.i_min = i_min
        self.i_max = i_max

        func = lambda x: clip(x, i_min, i_max)
        super(ClipIntensityd, self).__init__(keys=keys, func=func)
