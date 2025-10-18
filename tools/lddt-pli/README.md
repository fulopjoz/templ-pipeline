## LDDT-PLI scripts and notebooks
as an additional evaluation metric, post hoc we evaluated protein ligand poses using lDDT-PLI.
The scripts and notebooks we used for that are made available in this folder. 
This presupposes to user pointing the scripts towards the folders with results in them and with data like PDBBind and posebusters.

Outputs are also provided in the output/ folder.

The script utilizes the provided docker image and is used as follows:
```
./calc_lddt.sh input.csv output.csv
```
with `input.csv` a file of the form
```
apo,ref,lig
apo.pdb,ref_molecule.sdf,predicted_pose.sdf
...,...,...
```
