import json

datasets = {
    "openhermes": 243000,
    "wikitext": 37000,
    "openorca": 2940000,
    "arc": 1120,
    "metamath": 395000,
    "gsm8k": 7470,
    "alpaca": 52000,
    "codefeedback": 65400,
    "wizardlm": 70000
}

epoch_data = {
    "epoch_0.03": 0.22500073909759521,
    "epoch_0.06": 0.22500033676624298,
    "epoch_0.1": 0.22499977052211761,
    "epoch_0.13": 0.22499914467334747,
    "epoch_0.16": 0.2249983698129654,
    "epoch_0.19": 0.22499746084213257,
    "epoch_0.22": 0.22499646246433258,
    "epoch_0.26": 0.22499532997608185,
    "epoch_0.29": 0.22499392926692963,
    "epoch_0.32": 0.22499236464500427,
    "epoch_0.35": 0.2249905914068222,
    "epoch_0.38": 0.22498859465122223,
    "epoch_0.42": 0.22498641908168793,
    "epoch_0.45": 0.2249838262796402,
    "epoch_0.48": 0.22498087584972382,
    "epoch_0.51": 0.22497768700122833,
    "epoch_0.54": 0.22497421503067017,
    "epoch_0.58": 0.22497014701366425,
    "epoch_0.61": 0.22496575117111206,
    "epoch_0.64": 0.2249610871076584,
    "epoch_0.67": 0.2249559760093689,
    "epoch_0.7": 0.22495026886463165,
    "epoch_0.74": 0.22494402527809143,
    "epoch_0.77": 0.2249373197555542,
    "epoch_0.8": 0.22493039071559906,
    "epoch_0.83": 0.2249227911233902,
    "epoch_0.86": 0.22491468489170074,
    "epoch_0.9": 0.2249060720205307,
    "epoch_0.93": 0.22489720582962036,
    "epoch_0.96": 0.2248879075050354,
    "epoch_0.99": 0.22487789392471313,
    "epoch_1.0": 0.22487522661685944
}

available_epochs = [float(k.split('_')[1]) for k in epoch_data.keys()]

def find_nearest_epoch(dataset_size, epochs):
    target = dataset_size / 100000
    return min(epochs, key=lambda x: abs(x - target))

result = {}
for name, size in datasets.items():
    result[name] = find_nearest_epoch(size, available_epochs)

json_output = json.dumps(result, indent=4)

with open('dataset_epochs.json', 'w') as f:
    f.write(json_output)