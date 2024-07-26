{
    tmux_name=$(basename "${0%.*}")
    objects=(hotdog teapot head bear cabbage dog realchair)
    exp_names=(synthetic synthetic synthetic real real real real)
    gpus=(0 1 2 3 4 5 6)
    N=${#objects[@]}
    conda_name=nerfemitter

    tmux new-session -d -s ${tmux_name}
    for((i=0;i<N;i++)); do
        object=${objects[i]}
        exp_name=${exp_names[i]}
        gpu=${gpus[i]}
        
        ((cur_idx=i+1))
        tmux new-window -t ${tmux_name}:${cur_idx} -n ${object}
        tmux send-keys -t ${tmux_name}:${cur_idx} "export CUDA_VISIBLE_DEVICES=${gpu}" ENTER
        tmux send-keys -t ${tmux_name}:${cur_idx} "conda activate ${conda_name}" ENTER
        tmux send-keys -t ${tmux_name}:${cur_idx} "bash scripts/${exp_name}/baseline/run_${object}.sh" ENTER
        tmux send-keys -t ${tmux_name}:${cur_idx} "exit" ENTER
    done
    
    desired_count=1
    poll_interval=1  # Adjust this interval as needed

    while true; do
        window_count=$(tmux list-windows -t "${tmux_name}" | wc -l)

        if [ "$window_count" -eq "$desired_count" ]; then
            break
        else
            sleep "$poll_interval"
        fi
    done

    tmux kill-session -t ${tmux_name}
    exit
}