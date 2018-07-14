# custom label encoder with support for new labels
# idea from https://github.com/scikit-learn/scikit-learn/pull/3483

import operator
import warnings
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d


class CustomLabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, new_labels="raise"):
        self.new_labels = new_labels
        self.new_label_mapping_ = {}
        self.fit_labels_ = []

    def _check_fitted(self):
        if len(self.fit_labels_) == 0:
            raise ValueError("LabelEncoder was not fitted yet.")

    def get_classes(self):
        # If we've seen updates, include them in the order they were added.
        if len(self.new_label_mapping_) > 0:
            # Sort the post-fit time labels to return into the class array.
            sorted_new, _ = zip(*sorted(self.new_label_mapping_.items(),
                                        key=operator.itemgetter(1)))
            return np.append(self.fit_labels_, sorted_new)
        else:
            return self.fit_labels_

    def set_classes(self, classes):
        self.fit_labels_ = classes

    classes_ = property(get_classes, set_classes)

    def fit(self, y):
        # Check new_labels parameter
        if self.new_labels not in ["update", "raise"] and \
                type(self.new_labels) not in [int]:
            # Raise on invalid argument.
            raise ValueError("Value of argument `new_labels`={0} "
                             "is unknown and not integer."
                             .format(self.new_labels))

        y = column_or_1d(y, warn=True)
        self.fit_labels_ = np.unique(y)
        return self

    def fit_transform(self, y):
        # Check new_labels parameter
        if self.new_labels not in ["update", "raise"] and \
                type(self.new_labels) not in [int]:
            # Raise on invalid argument.
            raise ValueError("Value of argument `new_labels`={0} "
                             "is unknown and not integer."
                             .format(self.new_labels))

        y = column_or_1d(y, warn=True)
        self.fit_labels_, y = np.unique(y, return_inverse=True)
        return y

    def transform(self, y):
        self._check_fitted()
        classes = np.unique(y)
        if len(np.intersect1d(classes, self.get_classes())) < len(classes):
            # Get the new classes
            diff_fit = np.setdiff1d(classes, self.fit_labels_)
            diff_new = np.setdiff1d(classes, self.get_classes())

            # Create copy of array and return
            y = np.array(y)

            # If we are mapping new labels, get "new" ID and change in copy.
            if self.new_labels == "update":
                # Update the new label mapping
                next_label = len(self.get_classes())
                self.new_label_mapping_.update(dict(zip(diff_new,
                                                        range(next_label,
                                                              next_label +
                                                              len(diff_new)))))

                # Find entries with new labels
                missing_mask = np.in1d(y, diff_fit)

                # Populate return array properly by mask and return
                out = np.searchsorted(self.fit_labels_, y)
                out[missing_mask] = [[self.new_label_mapping_[value[0]]]
                                     for value in y[missing_mask]]
                return out
            elif type(self.new_labels) in [int]:
                # Update the new label mapping
                self.new_label_mapping_.update(dict(zip(diff_new,
                                                        [self.new_labels]
                                                        * len(diff_new))))

                # Find entries with new labels
                missing_mask = np.in1d(y, diff_fit)

                # Populate return array properly by mask and return
                out = np.searchsorted(self.fit_labels_, y)
                out[missing_mask] = self.new_labels
                return out
            elif self.new_labels == "raise":
                # Return ValueError, original behavior.
                raise ValueError("y contains new labels: %s" % str(diff_fit))
            else:
                # Raise on invalid argument.
                raise ValueError("Value of argument `new_labels`={0} "
                                 "is unknown.".format(self.new_labels))

        return np.searchsorted(self.fit_labels_, y)

    def inverse_transform(self, y):
        self._check_fitted()

        if type(self.new_labels) in [int]:
            warnings.warn('When ``new_labels`` uses an integer '
                          're-labeling strategy, the ``inverse_transform`` '
                          'is not necessarily one-to-one mapping; any '
                          'labels not present during initial ``fit`` will '
                          'not be mapped.',
                          UserWarning)

        y = np.asarray(y)
        try:
            return self.get_classes()[y]
        except IndexError:
            # Raise exception
            num_classes = len(self.get_classes())
            raise ValueError("Classes were passed to ``inverse_transform`` "
                             "with integer new_labels strategy ``fit``-time: "
                             "{0}"
                             .format(np.setdiff1d(y, range(num_classes))))