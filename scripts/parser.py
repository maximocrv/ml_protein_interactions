test_files = ["5E6P_A_B_RB42A.pdb", "5E6P_A_B_RB42E.pdb", "5E6P_A_B_YA468A.pdb", "5E9D_AB_CDE_AE98Y.pdb", "5E9D_AB_CDE_GD27E.pdb", "5E9D_AB_CDE_GD27E_AE98Y.pdb"]

for f in test_files:
    #split = f.split('_')
    chains = [[char for char in subchain] for subchain  in f.split('_')[1:3]]
    print(chains)
