1. sh train.sh <input_dir>

   for example, sh train.sh "./data"

2. sh predict.sh <input_dir> <new_test_data_path> <output_dir>

   for example, sh predict.sh "./data" "./data/test_features.csv" "./prediction"

TODOS:

> "drop features with the top 50 standard deviation 
   in ctl_samples or p>0.1 in scipy.stats.ks_2sample 
   between not_ctl_samples and ctl_samples (original features)"