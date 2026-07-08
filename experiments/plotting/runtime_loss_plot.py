import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams.update({'font.size': 11})

# Directory holding one subfolder of TensorBoard event files per sweep
# (e.g. experiments/results/tensorboard/sweep_22/). Override with the
# BALORA_EVENTS_DIR environment variable.
EVENTS_DIR = os.environ.get(
    "BALORA_EVENTS_DIR",
    os.path.join("..", "results", "tensorboard"),
)

plot_train_loss = False

file_name = "llama_wikitext" # "llama_gsm8k", "llama_wikitext", "custom", "qwen_metamath"
# figsize = (2.3,1.9)
figsize = (3.7, 2.4) # llama wikitext
# figsize = (4.3, 3.)  # llama gsm8k

if file_name == "custom":
    sweep_list = ["sweep_23_lorarite"]
    method_names = ["LoRA-RITE"]
    stop_time = None
    plot_start_time = None
    ylim = None

elif file_name == "OpenHermes":
    sweep_list = ["sweep_100", "sweep_101"]
    method_names = ["LoRA", "BaLoRA"]
    stop_time = None
    plot_start_time = None
    ylim = None

elif file_name == "CodeFeedback":
    sweep_list = ["sweep_10001", "sweep_10000"]
    method_names = ["LoRA", "BaLoRA"]
    stop_time = None
    plot_start_time = None
    ylim = None

elif file_name == "llama_metamath":
    sweep_list = ["sweep_40", "sweep_41", "sweep_43", "sweep_42", "sweep_45"]
    method_names = ["LoRA", "BaLoRA", "OLoRA", "BaLoRA-GA", "LoRA-GA"]
    stop_time = None
    plot_start_time = None

elif file_name == "qwen_metamath":
    sweep_list = ["sweep_70", "sweep_71", "sweep_73"] #, "sweep_75", "sweep_74"
    method_names = ["LoRA", "BaLoRA", "OLoRA"] #, "LoRA-GA", "DoRA"
    stop_time = None
    plot_start_time = None
    ylim = None

elif file_name == "llama_gsm8k":
    sweep_list = ["sweep_25", "sweep_26",   "sweep_32", "sweep_29", "sweep_31"] #"sweep_30",
    method_names = ["LoRA", "BaLoRA",  "OLoRA", "LoRA-GA", "DoRA"] # "BaLoRA-GA",
    stop_time = 150.
    plot_start_time = 10.
    ylim = [0.49, 0.65]

elif file_name == "WizardLM":
    sweep_list = ["sweep_1000", "sweep_1001"]
    method_names = ["LoRA", "BaLoRA"]
    stop_time = None
    plot_start_time = None
    ylim = None

elif file_name == "llama_wikitext":
    sweep_list = ["sweep_23", "sweep_22", "sweep_17", "sweep_18", "sweep_21", "sweep_23_lorarite", "sweep_23_reflora"] #, "sweep_16"
    method_names = ["LoRA", "BaLoRA", "OLoRA", "LoRA-GA", "DoRA", "LoRA-RITE", "RefLoRA"] #, "BaLoRA-GA"
    stop_time = None #1000
    plot_start_time = 0
    ylim = None

elif file_name == "gpt2_wikitext":
    sweep_list = ["sweep_1", "sweep_2", "sweep_4"]
    method_names = ["LoRA", "BaLoRA", "DoRA"]
    stop_time = None
    plot_start_time = None
    ylim = None

assert len(sweep_list) == len(method_names)

plt.figure(figsize=figsize)

color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

for trajectory, sweep_name in zip(method_names, sweep_list):
    test_losses_to_average = []
    train_losses_to_average = []

    for append_to_name in ["", "_bis", "_tris"]:
        print('Processing sweep:', sweep_name + append_to_name)
        try:
            event_name = os.listdir(os.path.join(EVENTS_DIR, sweep_name + append_to_name))[-1]
            event_file = os.path.join(EVENTS_DIR, sweep_name + append_to_name, event_name)
        except:
            pass

        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()

        # print(ea.Tags())

        ## test loss
        eval_loss_event = ea.Scalars("eval/loss")

        eval_runtime_event = ea.Scalars("eval/runtime")

        # Compute runtime relative to first event
        start_time = eval_runtime_event[0].wall_time
        # print([e.wall_time for e in events])
        test_times = [(e.wall_time - start_time) / 60.0 for e in eval_runtime_event]  # minutes
        test_losses = [e.value for e in eval_loss_event]
        print("len test losses:", len(test_losses))
        test_losses_to_average.append(test_losses)




        ## train loss
        if plot_train_loss:
            events = ea.Scalars("train/loss")

            steps = [e.step for e in events]
            eval_runtime_event = ea.Scalars("eval/runtime")

            # Compute runtime relative to first event
            start_time = eval_runtime_event[0].wall_time
            end_time = eval_runtime_event[-1].wall_time
            # print([e.wall_time for e in events])
            # times = [(e.wall_time - start_time) / 60.0 for e in eval_runtime_event]  # minutes
            train_losses = [e.value for e in events]
            print(train_losses)
            print("len train losses:", len(train_losses))
            train_times = np.linspace(0, (end_time - start_time) / 60.0, len(train_losses))
            train_losses_to_average.append(train_losses)


    if len(test_losses_to_average) > 1:
        test_losses = np.mean(np.array(test_losses_to_average), axis=0)
        if sweep_name == "sweep_18":
            # test_losses  = np.insert(test_losses, 0, test_losses[0])
            test_times = np.array(test_times) + 500
            # test_times = np.insert(test_times, 0, 0)
            # print(test_times)
        if sweep_name == "sweep_29":
            test_times = np.array(test_times) + 100
    else:
        test_losses = test_losses_to_average[0]
        # if sweep_name == "sweep_18":
        #     test_losses.insert(0, test_losses[0])
        #     print(test_times)
        #     test_times = test_times + 500
        #     print(test_times)
        #     test_times.insert(0, 0)
    plt.plot(test_times, test_losses, label=trajectory, ls='-', c=color_list[method_names.index(trajectory)], linewidth=0.4)

    if plot_train_loss:
        if len(train_losses_to_average) > 1:
            train_losses = np.mean(np.array(train_losses_to_average), axis=0)
        else:
            train_losses = train_losses_to_average[0]
        plt.plot(train_times, train_losses, ls='--', c=color_list[method_names.index(trajectory)], linewidth=0.4)
        # plt.plot([e.step for e in events], losses, label="Train loss " + trajectory)

plt.xlabel("Runtime (minutes)")
plt.ylabel("Loss")
if stop_time is not None and plot_start_time is not None:
    plt.xlim(left=plot_start_time, right=stop_time)
elif plot_start_time is not None:
    plt.xlim(left=0, right=stop_time)
elif stop_time is not None:
    plt.xlim(right=stop_time)
if ylim is not None:
    plt.ylim(bottom=ylim[0], top=ylim[1])
# plt.title(file_name)
plt.legend(handlelength=1)
plt.grid(True)
# plt.xscale("log")
# plt.yscale("log")
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig(f'figures/{file_name}_runtime_plot_{plot_train_loss}.pdf')