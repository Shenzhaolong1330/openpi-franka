import pyarrow.parquet as pq
import pandas as pd

path = '/vepfs-mlp2/c20250510/250303034/workspace/data/pick_and_place_all_merged_last_dance_v3.0/meta/tasks.parquet'
# ['task_index']
# path = '/vepfs-mlp2/c20250510/250303034/workspace/data/pick_and_place_all_merged_last_dance_v3.0/meta/episodes/chunk-000/file-000.parquet'
# ['episode_index', 'tasks', 'length', 'data/chunk_index', 'data/file_index', 'dataset_from_index', 'dataset_to_index', 'videos/observation.images.wrist_image/chunk_index', 'videos/observation.images.wrist_image/file_index', 'videos/observation.images.wrist_image/from_timestamp', 'videos/observation.images.wrist_image/to_timestamp', 'videos/observation.images.exterior_image/chunk_index', 'videos/observation.images.exterior_image/file_index', 'videos/observation.images.exterior_image/from_timestamp', 'videos/observation.images.exterior_image/to_timestamp', 'stats/action/min', 'stats/action/max', 'stats/action/mean', 'stats/action/std', 'stats/action/count', 'stats/action/q01', 'stats/action/q10', 'stats/action/q50', 'stats/action/q90', 'stats/action/q99', 'stats/observation.state/min', 'stats/observation.state/max', 'stats/observation.state/mean', 'stats/observation.state/std', 'stats/observation.state/count', 'stats/observation.state/q01', 'stats/observation.state/q10', 'stats/observation.state/q50', 'stats/observation.state/q90', 'stats/observation.state/q99', 'stats/observation.images.wrist_image/min', 'stats/observation.images.wrist_image/max', 'stats/observation.images.wrist_image/mean', 'stats/observation.images.wrist_image/std', 'stats/observation.images.wrist_image/count', 'stats/observation.images.wrist_image/q01', 'stats/observation.images.wrist_image/q10', 'stats/observation.images.wrist_image/q50', 'stats/observation.images.wrist_image/q90', 'stats/observation.images.wrist_image/q99', 'stats/observation.images.exterior_image/min', 'stats/observation.images.exterior_image/max', 'stats/observation.images.exterior_image/mean', 'stats/observation.images.exterior_image/std', 'stats/observation.images.exterior_image/count', 'stats/observation.images.exterior_image/q01', 'stats/observation.images.exterior_image/q10', 'stats/observation.images.exterior_image/q50', 'stats/observation.images.exterior_image/q90', 'stats/observation.images.exterior_image/q99', 'stats/timestamp/min', 'stats/timestamp/max', 'stats/timestamp/mean', 'stats/timestamp/std', 'stats/timestamp/count', 'stats/timestamp/q01', 'stats/timestamp/q10', 'stats/timestamp/q50', 'stats/timestamp/q90', 'stats/timestamp/q99', 'stats/frame_index/min', 'stats/frame_index/max', 'stats/frame_index/mean', 'stats/frame_index/std', 'stats/frame_index/count', 'stats/frame_index/q01', 'stats/frame_index/q10', 'stats/frame_index/q50', 'stats/frame_index/q90', 'stats/frame_index/q99', 'stats/episode_index/min', 'stats/episode_index/max', 'stats/episode_index/mean', 'stats/episode_index/std', 'stats/episode_index/count', 'stats/episode_index/q01', 'stats/episode_index/q10', 'stats/episode_index/q50', 'stats/episode_index/q90', 'stats/episode_index/q99', 'stats/index/min', 'stats/index/max', 'stats/index/mean', 'stats/index/std', 'stats/index/count', 'stats/index/q01', 'stats/index/q10', 'stats/index/q50', 'stats/index/q90', 'stats/index/q99', 'stats/task_index/min', 'stats/task_index/max', 'stats/task_index/mean', 'stats/task_index/std', 'stats/task_index/count', 'stats/task_index/q01', 'stats/task_index/q10', 'stats/task_index/q50', 'stats/task_index/q90', 'stats/task_index/q99', 'meta/episodes/chunk_index', 'meta/episodes/file_index']
# path = '/vepfs-mlp2/c20250510/250303034/workspace/data/pick_and_place_all_merged_last_dance_v3.0/data/chunk-000/file-000.parquet'
# ['action', 'observation.state', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']
table = pq.read_table(path)
df = table.to_pandas()

print('=== SCHEMA ===')
print(table.schema)
print()
print('=== SHAPE ===')
print(df.shape)
print()
print('=== COLUMNS ===')
print(df.columns.tolist())
print()
print('=== DATA (first row) ===')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 100)
print(df.head(1).to_string())