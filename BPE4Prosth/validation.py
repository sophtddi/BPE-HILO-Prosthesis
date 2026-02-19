from core_function import reload_model
import os 
from config import save_dir, explo_file, n_queries, dim
from core_function import translate_param
import numpy as np
from botorch.optim import optimize_acqf
import torch
import random
from botorch.acquisition import UpperConfidenceBound

torch.manual_seed(12)

file_path = os.path.join(save_dir, f"{4}_final_modeldata.pt")
restored_model, likelihood, checkpoint = reload_model(file_path)
best_estimate_1 = checkpoint["best_point"] # [49.1, 1.4, -46.9, 0.4]
challenger_to_estimate1 = torch.rand(3, dim) 
print(np.round(translate_param(best_estimate_1), 1).tolist())


file_path = os.path.join(save_dir, f"{2}_final_modeldata.pt")
restored_model, likelihood, checkpoint = reload_model(file_path)
best_estimate_2 = checkpoint["best_point"] # [40.0, 1.1, -41.9, 0.4]
challenger_to_estimate2 = torch.rand(3, dim)
print(np.round(translate_param(best_estimate_2), 1).tolist())

file_path = os.path.join(save_dir, f"{3}_final_modeldata.pt")
restored_model, likelihood, checkpoint = reload_model(file_path)
best_estimate_3 = checkpoint["best_point"] # [41.7, 1.5, -51.8, 0.4]
challenger_to_estimate3 = torch.rand(3, dim)
print(np.round(translate_param(best_estimate_3), 1).tolist())



pairs_1 = [(best_estimate_1, challenger_to_estimate1[i]) for i in range(3)]
pairs_2 = [(best_estimate_2, challenger_to_estimate2[i]) for i in range(3)]
pairs_3 = [(best_estimate_3, challenger_to_estimate3[i]) for i in range(3)]


def print_pairs(pairs, name):
    print(f"\n===== {name} =====")
    for i, (best, challenger) in enumerate(pairs):
        challenger_np = challenger.detach().cpu().numpy()

        translated_best = translate_param(best)
        translated_challenger = translate_param(challenger_np)

        print(f"\nPair {i+1}:")
        print("  best_estimate      =", translated_best)
        print("  challenger_estimate=", translated_challenger)

print_pairs(pairs_1, "PAIRS 1")
print_pairs(pairs_2, "PAIRS 2")
print_pairs(pairs_3, "PAIRS 3")

all_pairs = pairs_1 + pairs_2 + pairs_3     # liste de 9 paires
random.shuffle(all_pairs)   

def print_shuffled_pairs(pairs):
    print("\n===== PAIRES MÉLANGÉES =====")
    for idx, (best, challenger) in enumerate(pairs):
        
        challenger_np = challenger.detach().cpu().numpy()

        translated_best = translate_param(best)
        translated_challenger = translate_param(challenger_np)

        print(f"\nPair {idx+1}:")
        print("  best_estimate      =", translated_best)
        print("  challenger_estimate=", translated_challenger)

print_shuffled_pairs(all_pairs)


print('\nVALIDATION')

def validation_test(best_translate, challenger):
    bit = random.randint(0, 1)

    challenger_translated = translate_param(challenger)

    if bit == 1:
        print(f"Send option 1: {best_translate.tolist()}")
        print(f"Send point 2: {challenger_translated.tolist()}")

        pair_np = np.stack([best_translate, challenger_translated])
    else:
        print(f"Send point 1: {challenger_translated.tolist()}")
        print(f"Send option 2: {best_translate.tolist()}")

        pair_np = np.stack([challenger_translated, best_translate])

    # conversion propre en torch
    pair = torch.from_numpy(pair_np).double()

    while True:
        choice = input("Preference? (0 = None, 1 = Option 1 or 2 = Option 2): ")
        if choice in ['0', '1', '2']:
            return pair, int(choice)
        print("Enter 0, 1 or 2.")


results = []

for i, (best, challenger) in enumerate(all_pairs, start=1):
    print(f"\n===== VALIDATION {i} =====")

    best_translate = translate_param(best)
    challenger_np = challenger.detach().cpu().numpy()

    # Appel de la fonction de validation
    pair, choice = validation_test(best_translate, challenger_np)

    # Stocker le résultat
    results.append((best, challenger, pair, choice))



save_path = "results_validation.pt"
torch.save(results, save_path)
print(f"Saved results to {save_path}")

loaded_results = torch.load("results_validation.pt")
print("Reloaded", len(loaded_results), "entries")

for i, (pair, choice) in enumerate(loaded_results, start=1):
    print(f"\n===== VALIDATION {i} =====")
    
    option1 = pair[0].numpy()  # first option
    option2 = pair[1].numpy()  # second option

    print("Option 1:", option1.tolist())
    print("Option 2:", option2.tolist())
    print("User preference:", choice)

# for i in range(n_verif):                                        # for n_verif iterations
#     challenger = torch.rand(dim)  # tensor de shape (dim,) avec valeurs entre 0 et 1
#     pair, choice = validation_test(final_pref_estimate, challenger, i)     # ask the user its preference between current_best and challenger in a random order
#     print(f'User chose option {choice}, the true preference was {final_pref_estimate}')

#     save_model(best_config=final_pref_estimate, option1=pair[0], option2=pair[1], pref=choice,
#                 train_x=None, train_comp=None,  visited_pairs=None,
#                   model=None, likelihood=None, mll=None,
#                   save_dir=save_dir, test_id=test_id, i=f'_validation_{i}_')


# policy = UpperConfidenceBound(model=restored_model, beta=10000)
# candidate, _ = optimize_acqf(
#             policy,
#             bounds=torch.stack([torch.zeros(dim), torch.ones(dim)]),
#             q=1,                     
#             num_restarts=30,
#             raw_samples=512,
#         )
# print(np.round(translate_param(candidate), 1).tolist())