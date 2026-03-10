from openbabel import pybel
import os
import numpy as np
import torch
from torch_geometric.data import Data
from tfbio.data import Featurizer
from itertools import combinations


# ===================== 基础工具 =====================

def read_molecule(file, fmt):
    try:
        return next(pybel.readfile(fmt, file))
    except:
        raise IOError(f"Cannot read {file}")


featurizer = Featurizer()


# ===================== 8 Å 蛋白原子筛选（关键修正） =====================

def select_protein_atoms_within_8A(
    protein,
    protein_coords,
    protein_feats,
    ligand_coords,
    cutoff=8.0
):
    sel_atoms = []
    sel_coords = []
    sel_feats = []

    heavy_atoms = [a for a in protein if a.atomicnum != 1]

    for atom, coord, feat in zip(heavy_atoms, protein_coords, protein_feats):
        dists = np.linalg.norm(ligand_coords - coord, axis=1)
        if np.any(dists <= cutoff):
            sel_atoms.append(atom)
            sel_coords.append(coord)
            sel_feats.append(feat)

    return sel_atoms, np.array(sel_coords), np.array(sel_feats)

VDW_RADIUS = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
}

def add_vdw_radius_feature(feats19, atoms):
    """
    feats19: (N, 19) numpy array
    atoms:   list of pybel.Atom (length N)

    return:
        feats20: (N, 20) numpy array
    """
    feats20 = []

    for feat, atom in zip(feats19, atoms):
        symbol = atom.type.strip().capitalize()  # 如 'C', 'N', 'O'
        vdw = VDW_RADIUS.get(symbol, 1.5)
        feats20.append(np.concatenate([feat, [vdw]]))

    return np.array(feats20)

# ===================== 边特征相关 =====================

def bond_type_bits(bond):
    bits = [0, 0, 0, 0, 0]  # single, double, triple, aromatic, ring

    if bond is None:
        return bits

    order = bond.GetBondOrder()
    if order == 1:
        bits[0] = 1
    elif order == 2:
        bits[1] = 1
    elif order == 3:
        bits[2] = 1

    if bond.IsAromatic():
        bits[3] = 1
    if bond.IsInRing():
        bits[4] = 1

    return bits


def distance_bits(dist, is_covalent=False):
    if is_covalent:
        return [1, 0, 0]

    if 2.0 <= dist < 4.0:
        return [1, 0, 0]
    elif 4.0 <= dist < 6.0:
        return [0, 1, 0]
    elif 6.0 <= dist < 8.0:
        return [0, 0, 1]
    else:
        return [0, 0, 0]


def build_edge_index_and_attr(atoms, coords, is_ligand, cutoff=8.0):
    edge_index = []
    edge_attr = []

    N = len(atoms)

    for i, j in combinations(range(N), 2):
        atom_i = atoms[i]
        atom_j = atoms[j]

        dist = np.linalg.norm(coords[i] - coords[j])

        # 情况 1：共价键
        bond = atom_i.OBAtom.GetBond(atom_j.OBAtom)
        if bond is not None:
            b_bits = bond_type_bits(bond)
            d_bits = distance_bits(dist, is_covalent=True)

        # 情况 2：配体-蛋白非共价
        elif is_ligand[i] != is_ligand[j] and dist < cutoff:
            b_bits = [0, 0, 0, 0, 0]
            d_bits = distance_bits(dist)

        else:
            continue

        attr = b_bits + d_bits

        edge_index.append([i, j])
        edge_attr.append(attr)

        edge_index.append([j, i])
        edge_attr.append(attr)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return edge_index, edge_attr


# ===================== 主函数 =====================

def build_graph_from_pdbbind(
    ligand_file,
    protein_file,
    ligand_fmt="mol2",
    protein_fmt="pdb",
):
    ligand = read_molecule(ligand_file, ligand_fmt)
    protein = read_molecule(protein_file, protein_fmt)

    # Featurizer 输出
    lig_coords, lig_feats = featurizer.get_features(ligand, molcode=1)
    pro_coords, pro_feats = featurizer.get_features(protein, molcode=-1)

    # 原子对象（与 Featurizer 顺序一致）
    lig_atoms = [a for a in ligand if a.atomicnum != 1]

    pro_atoms_sel, pro_coords_sel, pro_feats_sel = select_protein_atoms_within_8A(
        protein,
        pro_coords,
        pro_feats,
        lig_coords,
        cutoff=8.0
    )

    # 拼接
    atoms = lig_atoms + pro_atoms_sel
    coords = np.concatenate([lig_coords, pro_coords_sel], axis=0)
    feats19 = np.concatenate([lig_feats, pro_feats_sel], axis=0)

    # ★ 新增：19 → 20 维
    feats20 = add_vdw_radius_feature(feats19, atoms)

    is_ligand = [True] * len(lig_atoms) + [False] * len(pro_atoms_sel)

    x = torch.tensor(feats20, dtype=torch.float)

    edge_index, edge_attr = build_edge_index_and_attr(
        atoms, coords, is_ligand, cutoff=8.0
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    return data


# # ===================== 测试 =====================
#
# if __name__ == "__main__":
#     base_dir = r"C:\Users\86136\Desktop\bishe\1a0q"
#
#     lig_file = os.path.join(base_dir, "1a0q_ligand.mol2")
#     pro_file = os.path.join(base_dir, "1a0q_protein.pdb")
#
#     g = build_graph_from_pdbbind(lig_file, pro_file)
#     print(g)

BASE_DIR = r"C:\Users\86136\Desktop\bishe\PDBbind\refined-set"
OUTPUT_DIR = r"C:\Users\86136\Desktop\bishe\graph_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读取 affinity
index_file = r"C:\Users\86136\Desktop\bishe\PDBbind\INDEX_refined_data.2020"

affinity_dict = {}

with open(index_file) as f:
    for line in f:
        if line.startswith("#"):
            continue

        parts = line.split()
        pdbid = parts[0]
        affinity = float(parts[3])

        affinity_dict[pdbid] = affinity


pdb_ids = os.listdir(BASE_DIR)

print("Total complexes:", len(pdb_ids))


for pdbid in tqdm(pdb_ids):

    folder = os.path.join(BASE_DIR, pdbid)

    ligand_file = os.path.join(folder, f"{pdbid}_ligand.mol2")
    protein_file = os.path.join(folder, f"{pdbid}_protein.pdb")

    try:

        graph = build_graph_from_pdbbind(
            ligand_file,
            protein_file
        )

        # 加入 label
        if pdbid in affinity_dict:
            graph.y = torch.tensor([affinity_dict[pdbid]], dtype=torch.float)

        save_path = os.path.join(OUTPUT_DIR, f"{pdbid}.pt")

        torch.save(graph, save_path)

    except Exception as e:

        print(f"Skip {pdbid}: {e}")