import pandas as pd
import os
import matplotlib.pyplot as plt


def _filter_run(df, run_id=None):
    if 'Run_ID' not in df.columns:
        return df, run_id

    run_ids = df['Run_ID'].dropna().astype(str)
    if run_ids.empty:
        return df, run_id

    if run_id is None:
        run_id = run_ids.iloc[-1]
        print(f"Using Run_ID={run_id}")

    filtered_df = df[df['Run_ID'].astype(str) == str(run_id)].copy()
    return filtered_df, str(run_id)


def plot_cumulative_reward(csv_file, run_id=None):
    df = pd.read_csv(csv_file)
    df, run_id = _filter_run(df, run_id=run_id)

    df['Computed_Cumulative_Rewards'] = df.groupby('Episode')['Rewards'].cumsum()
    final_rewards = df.groupby('Episode').last()

    plt.figure(figsize=(10, 5))
    plt.plot(final_rewards.index, final_rewards['Computed_Cumulative_Rewards'], marker='o', linestyle='-')

    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    title = "Cumulative Reward per Episode"
    if run_id is not None:
        title = f"{title} ({run_id})"
    plt.title(title)
    plt.grid()
    plt.show()


def plot_timestep_reward(csv_file, episode_num, run_id=None):
    df = pd.read_csv(csv_file)
    df, run_id = _filter_run(df, run_id=run_id)
    df_episode = df[df['Episode'] == episode_num]

    plt.figure(figsize=(10, 5))
    plt.plot(df_episode['Timestep'], df_episode['Rewards'], marker='o', linestyle='-')

    plt.xlabel("Timestep")
    plt.ylabel("Rewards")
    title = f"reward per timestep for episode {episode_num}"
    if run_id is not None:
        title = f"{title} ({run_id})"
    plt.title(title)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    file_names = ['2026_04_16Rewards_DesignCycle0_mixed_terrain']  # Replace with your actual file names
    for i in range(51):
        for file_path in file_names:
            counter = 4
            df = pd.read_csv(file_path, delimiter='\t')  # Change delimiter if needed
            df.to_csv(f'output_{counter}.csv', index=False)

            designs = ['output_4.csv']

            with open(designs[0], 'r') as file:
                filedata = file.read()

            filedata = filedata.replace('"', '')
            with open(designs[0], 'w') as file:
                file.write(filedata)

            plot_timestep_reward('output_4.csv', episode_num=i)
            plot_cumulative_reward('output_4.csv')
