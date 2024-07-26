{
    objects=(bear cabbage dog realchair)
    for object in ${objects[@]}; do
        bash scripts/real/ours/run_${object}.sh
    done
    exit
}