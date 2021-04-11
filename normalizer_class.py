
from copy import copy
from os import error

class normalizer:

    def __init__(self) -> None:

        self.norm_dict = {}

        self.ndim = 0

    def fit(self, x):

        if x.ndim == 2:
            self.ndim = 2

            for c in x.columns:
                mean_val = x[c].mean()
                std_val = x[c].std()

                if std_val == 0.0:
                    std_val = 1.0

                self.norm_dict[c] = {
                    'mean' : mean_val,
                    'std' : std_val
                }
        elif x.ndim == 1:
            self.ndim = 1

            mean_val = x.mean()
            std_val = x.std()

            if std_val == 0.0:
                std_val = 1.0

            self.norm_dict = {
                'mean' : mean_val,
                'std' : std_val
            }
        else:
            raise ValueError('normalizer got bad ndim of {}'.format(x.ndim))


    def transform(self, x):

        x_new = copy(x)

        if self.ndim == 2 and x.ndim == 2:
            for c in x_new.columns:
                x_new[c] = (x_new[c] - self.norm_dict[c]['mean']) / self.norm_dict[c]['std']
        elif self.ndim == 1 and x.ndim == 1:
            x_new = (x_new - self.norm_dict['mean']) / self.norm_dict['std']
        else:
            raise ValueError('normalizer got bad ndim of {}, expected {}.'.format(x.ndim, self.ndim))

        return x_new

    def fit_transform(self, x):

        self.fit(x)

        x_new = self.transform(x)

        return x_new