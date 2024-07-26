{
  object="$1"
  method_name=sdf-gt-envmap
  timestamp=v1
  num_gpu=1
  exp_name=real
  envs=(lythwood ballroom cathedral_prerotate)
  # train, need to run sdf-nerfacto first to get occlusion images at outputs/${exp_name}/${object}/sdf-nerfacto/${timestamp}
  python nerfstudio/scripts/train.py ${method_name} \
    --experiment-name ${object} \
    --output-dir outputs/${exp_name} \
    --timestamp ${timestamp} \
    --machine.num-devices ${num_gpu} \
    --vis tensorboard \
    --pipeline.emitter-xml-path scenes/envmap/${object}/scene.xml \
    --pipeline.use_occlusion_image True \
    --pipeline.occlusion_load_dir outputs/${exp_name}/${object}/sdf-nerfacto/${timestamp} \
    --pipeline.datamanager.load-dir outputs/${exp_name}/${object}/sdf-nerfacto/${timestamp}/nerfstudio_models/ \
    --pipeline.datamanager.camera_optimizer.mode SO3xR3 \
    --pipeline.datamanager.camera_optimizer.optimizer.lr 0 \
    --pipeline.datamanager.cache_images no-cache \
    --pipeline.datamanager.images_on_gpu False \
    --pipeline.bbox_constraint False \
    --pipeline.hide-emitters True \
    --pipeline.load-voxel-path scenes/init_voxels/${object} \
    nerfstudio-data --data data/nerfstudio/${object} \
    --mi-data data/nerfstudio/${object}_cropped \
    --downscale_factor 4 \
    --mock-split-by-valid True
  # export mesh
  python nerfstudio/scripts/exporter.py mi-marching-cubes \
    --load-config outputs/${exp_name}/${object}/${method_name}/${timestamp}/config.yml \
    --output-dir results/${exp_name}/${object}/${method_name}/${timestamp}
  # relighting
  for env in ${envs[@]}; do
    python nerfstudio/scripts/render.py eval \
      --load-config outputs/${exp_name}/${object}/${method_name}/${timestamp}/config.yml \
      --output-path results/${exp_name}/${object}/${method_name}/${timestamp}/${env} \
      --output-format images \
      --mock_zero_rotation True \
      --eval-mode all \
      --image-format png \
      --use-mi-train True \
      --hide-emitters False \
      --emitter-xml-path scenes/emitters/${env}/scene.xml
  done
  # novel-view synthesis
  python nerfstudio/scripts/eval.py \
    --load-config outputs/${exp_name}/${object}/${method_name}/${timestamp}/config.yml \
    --render-output-path results/${exp_name}/${object}/${method_name}/${timestamp}/novel_view \
    --output-path results/${exp_name}/${object}/${method_name}/${timestamp}/novel_view/metric.json \
    --load-occlusion \
    --mi-data-split test \
    --override_eval_images_metrics True
  exit
}
