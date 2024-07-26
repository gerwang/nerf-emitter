{
    objects=(teapot head hotdog)
    for object in ${objects[@]}; do
        bash scripts/synthetic/ours/run_${object}.sh
    done
    exit
}