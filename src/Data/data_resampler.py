import logging
import numpy as np
from imblearn.over_sampling import SMOTE


class DataResampler:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def apply_smote(self, sampling_strategy="auto", random_state=42, k_neighbors=5):
        before_counts = np.bincount(self.y)
        logging.info(f"Class distribution before SMOTE: {before_counts}")

        try:
            # Apply SMOTE
            logging.info(
                f"Applying SMOTE with strategy: {sampling_strategy}, k_neighbors={k_neighbors}"
            )
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=random_state,
                k_neighbors=k_neighbors,
            )
            X_resampled, y_resampled = smote.fit_resample(self.x, self.y)

            # Log class distribution after SMOTE
            after_counts = np.bincount(y_resampled)
            logging.info(f"Class distribution after SMOTE: {after_counts}")
            # logging.info(f"Original training set size: {len(self.x)} samples")
            logging.info(f"Resampled training set size: {len(X_resampled)} samples")

            return X_resampled, y_resampled
        except ValueError as e:
            logging.warning(f"SMOTE failed: {e}. Using original data.")
            return self.x, self.y

    # def resample_data_with_adasyn(self):
    #     pass

    # def resample_data_with_random_oversampling(self):
    #     pass
