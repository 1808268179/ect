# ect
python train.py \
  --data-root /data/users/jw/data/different_camera/c1_train \
  --output-dir outputs/ectformer_c1 \
  --variant x1.0 \
  --epochs 100 \
  --batch-size 64 \
  --image-size 224 \
  --lr 1e-3 \
  --weight-decay 5e-2 \
  --label-smoothing 0.1
