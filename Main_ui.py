import streamlit as st
import subprocess
import os

st.set_page_config(page_title="RL Dashboard", layout="wide")
st.title("🤖 Reinforcement Learning with BabyAI")

# Create tabs correctly
tab0, tab1, tab2, tab3 = st.tabs(["Home", "Train RL", "Evaluate Model", "Visualize Agent"])

# ----- TAB 0: HOME -----
command_list = {
    "Manual Control": "python3 manual_control.py",
    "Q-learning_empty-8x8": "python3 Q_learning_empty_8x8.py",
    "Compare": "python3 compare_Q_vs_Astar.py",
    "Q-learning_dynamic_obs": "python3 Q_learning_dynamic_obs.py",
}

with tab0:
    st.header("🏠 Quick Actions")

    for label, command in command_list.items():
        if st.button(label):
            st.subheader("Running Command")
            st.code(command)
            with st.spinner(f"Executing `{label}`..."):
                try:
                    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    st.subheader("Output")
                    st.text(result.stdout)
                except Exception as e:
                    st.error(f"Error running command `{command}`: {str(e)}")


# ----- TAB 1: TRAIN RL -----
with tab1:
    st.header("Train PPO on BabyAI And Minigrid")

    with st.form("train_form"):
        env = st.text_input("Environment (REQUIRED)", key="train_env")
        model = st.text_input("Model name", key="train_model")
        seed = st.text_input("Seed", key="train_seed")
        log_interval = st.text_input("Log Interval", key="train_log_interval")
        save_interval = st.text_input("Save Interval", key="train_save_interval")
        procs = st.text_input("Processes", key="train_procs")
        frames = st.text_input("Total Frames", key="train_frames")

        epochs = st.text_input("PPO Epochs", key="train_epochs")
        batch_size = st.text_input("Batch Size", key="train_batch_size")
        frames_per_proc = st.text_input("Frames per Proc", key="train_frames_per_proc")
        discount = st.text_input("Discount Factor", key="train_discount")
        lr = st.text_input("Learning Rate", key="train_lr")
        gae_lambda = st.text_input("GAE Lambda", key="train_gae_lambda")
        entropy_coef = st.text_input("Entropy Coef", key="train_entropy_coef")
        value_loss_coef = st.text_input("Value Loss Coef", key="train_value_loss_coef")
        max_grad_norm = st.text_input("Max Grad Norm", key="train_max_grad_norm")
        optim_eps = st.text_input("Optim Eps", key="train_optim_eps")
        optim_alpha = st.text_input("Optim Alpha", key="train_optim_alpha")
        clip_eps = st.text_input("Clip Eps", key="train_clip_eps")
        recurrence = st.text_input("Recurrence", key="train_recurrence")
        text_flag = st.checkbox("Enable Text Input Model (GRU)", key="train_text")

        submitted = st.form_submit_button("Train")

    if submitted:
        cmd = ["python3", "ppo_babyai/scripts/train.py"]

        if env: cmd += ["--env", env]
        if model: cmd += ["--model", model]
        if seed: cmd += ["--seed", seed]
        if log_interval: cmd += ["--log-interval", log_interval]
        if save_interval: cmd += ["--save-interval", save_interval]
        if procs: cmd += ["--procs", procs]
        if frames: cmd += ["--frames", frames]

        if epochs: cmd += ["--epochs", epochs]
        if batch_size: cmd += ["--batch-size", batch_size]
        if frames_per_proc: cmd += ["--frames-per-proc", frames_per_proc]
        if discount: cmd += ["--discount", discount]
        if lr: cmd += ["--lr", lr]
        if gae_lambda: cmd += ["--gae-lambda", gae_lambda]
        if entropy_coef: cmd += ["--entropy-coef", entropy_coef]
        if value_loss_coef: cmd += ["--value-loss-coef", value_loss_coef]
        if max_grad_norm: cmd += ["--max-grad-norm", max_grad_norm]
        if optim_eps: cmd += ["--optim-eps", optim_eps]
        if optim_alpha: cmd += ["--optim-alpha", optim_alpha]
        if clip_eps: cmd += ["--clip-eps", clip_eps]
        if recurrence: cmd += ["--recurrence", recurrence]
        if text_flag: cmd.append("--text")

        command_str = " ".join(cmd)
        st.subheader("Running Command")
        st.code(command_str)

        with st.spinner("Training in progress..."):
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            st.subheader("Output")
            st.text(result.stdout)

# ----- TAB 2: EVALUATE MODEL -----
with tab2:
    st.header("Evaluate RL Trained Model")
    with st.form("eval_form"):
        eval_env = st.text_input("Environment (REQUIRED)", key="eval_env")
        eval_model = st.text_input("Model (REQUIRED)", key="eval_model")
        episodes = st.text_input("Number of Episodes", key="eval_episodes")
        seed = st.text_input("Seed", key="eval_seed")
        procs = st.text_input("Number of Processes", key="eval_procs")
        worst_episodes = st.text_input("Worst Episodes to Show", key="eval_worst_episodes")
        memory = st.checkbox("Use LSTM Memory", key="eval_memory")
        text = st.checkbox("Use GRU Text", key="eval_text")
        argmax = st.checkbox("Use Argmax Policy", key="eval_argmax")

        eval_submitted = st.form_submit_button("Evaluate")

    if eval_submitted:
        cmd = ["python3", "ppo_babyai/scripts/evaluate.py"]
        if eval_env: cmd += ["--env", eval_env]
        if eval_model: cmd += ["--model", eval_model]
        if episodes: cmd += ["--episodes", episodes]
        if seed: cmd += ["--seed", seed]
        if procs: cmd += ["--procs", procs]
        if worst_episodes: cmd += ["--worst-episodes-to-show", worst_episodes]
        if memory: cmd.append("--memory")
        if text: cmd.append("--text")
        if argmax: cmd.append("--argmax")

        command_str = " ".join(cmd)
        st.subheader("Running Command")
        st.code(command_str)

        with st.spinner("Evaluating model..."):
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            st.subheader("Output")
            st.text(result.stdout)

# ----- TAB 3: VISUALIZE AGENT -----
with tab3:
    st.header("Visualize Trained Agent")
    with st.form("viz_form"):
        viz_model = st.text_input("Model name to visualize", key="viz_model")
        viz_env = st.text_input("Environment to visualize on", key="viz_env")
        seed = st.text_input("Seed", key="viz_seed")
        shift = st.text_input("Shift (Initial resets)", key="viz_shift")
        pause = st.text_input("Pause between actions", key="viz_pause")
        episodes = st.text_input("Number of Episodes", key="viz_episodes")
        gif = st.text_input("GIF Output Filename", key="viz_gif")
        memory = st.checkbox("Use LSTM Memory", key="viz_memory")
        text = st.checkbox("Use GRU Text", key="viz_text")
        argmax = st.checkbox("Use Argmax Policy", key="viz_argmax")

        viz_submitted = st.form_submit_button("Visualize")

    if viz_submitted:
        cmd = ["python3", "ppo_babyai/scripts/visualize.py"]
        if viz_env: cmd += ["--env", viz_env]
        if viz_model: cmd += ["--model", viz_model]
        if seed: cmd += ["--seed", seed]
        if shift: cmd += ["--shift", shift]
        if pause: cmd += ["--pause", pause]
        if episodes: cmd += ["--episodes", episodes]
        if gif: cmd += ["--gif", gif]
        if memory: cmd.append("--memory")
        if text: cmd.append("--text")
        if argmax: cmd.append("--argmax")

        command_str = " ".join(cmd)
        st.subheader("Running Command")
        st.code(command_str)

        with st.spinner("Rendering agent..."):
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            st.subheader("Output")
            st.text(result.stdout)
