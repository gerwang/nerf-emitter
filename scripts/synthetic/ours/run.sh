{
  object="$1"
  emitter="$2"
  optconfig="$3"
  method_name=sdf-nerfacto
  timestamp=v1
  num_gpu=2
  exp_name=synthetic
  envs=(lythwood ballroom cathedral_prerotate)
  # train
  python nerfstudio/scripts/train.py ${method_name} \
    --experiment-name ${object}_${emitter} \
    --output-dir outputs/${exp_name} \
    --timestamp ${timestamp} \
    --vis tensorboard \
    --machine.num-devices ${num_gpu} \
    --pipeline.mi-opt-config-name ${optconfig} \
    --pipeline.load-voxel-path scenes/init_voxels/${object}/ \
    instant-ngp-data --data data/instant-ngp/${object}_in_${emitter}_rotated --load-mask False
  # export mesh
  python nerfstudio/scripts/exporter.py mi-marching-cubes \
    --load-config outputs/${exp_name}/${object}_${emitter}/${method_name}/${timestamp}/config.yml \
    --output-dir results/${exp_name}/${object}_${emitter}/${method_name}/${timestamp}
  # novel-view synthesis
  python nerfstudio/scripts/render.py eval \
    --load-config outputs/${exp_name}/${object}_${emitter}/${method_name}/${timestamp}/config.yml \
    --output-path results/${exp_name}/${object}_${emitter}/${method_name}/${timestamp}/novel_view \
    --spp 256 \
    --spp-per-batch 64 \
    --primal_spp_mult 1 \
    --machine.num-devices ${num_gpu} \
    --output-format images \
    --image-format exr \
    --denoise True
  # relighting
  for env in ${envs[@]}; do
    python nerfstudio/scripts/eval.py \
      --load-config outputs/${exp_name}/${object}_${emitter}/${method_name}/${timestamp}/config.yml \
      --render-output-path results/${exp_name}/${object}_${emitter}/${method_name}/${timestamp}/${env} \
      --output-path results/${exp_name}/${object}_${emitter}/${method_name}/${timestamp}/${env}/metric.json \
      --emitter-xml-path scenes/emitters/${env}/scene.xml \
      --test-data data/instant-ngp/relight_evaluate/${object}_in_${env}/ \
      --mock_zero_rotation True \
      --pre-mult-mask False \
      --eval-mode all \
      --load-mask True \
      --eval-use-mask True
  done
  exit
}
