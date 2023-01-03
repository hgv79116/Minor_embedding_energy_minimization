from dimod import BinaryQuadraticModel, ExactSolver, StructureComposite
from dimod.vartypes import SPIN
from dimod.binary import BinaryQuadraticModel
from neal import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
import dwave_networkx as dnx
from dwave.embedding.chain_strength import uniform_torque_compensation, scaled
from math import sqrt
import csv

EPS = 0.0001
CUTOFF = 15
RATIO = 0.22

def read_input(n): 
    h = {}
    J = {}
    input_arr = list(map(int, input().split()))
    h = dict(enumerate(input_arr))        
    for i in range(n): 
        for j in range(i + 1, n): 
            J[(i, j)] = int(input())

    return h, J

def get_my_chain_strength(n: int, h: dict, J: dict): 
    sum = {}
    for i in range(n): 
        sum[i] = abs(h[i] if i in h else 0)

    for i in range(0, n): 
        for j in range(i + 1, n): 
            sum[i] += abs(J[(i, j)])
            sum[j] += abs(J[(i, j)])

    chain_strength = max([RATIO * value for value in sum.values()])

    return chain_strength

def get_clique_embedding(n: int, h: dict, J: dict, chain_strength, row_base = 0, col_base = 0): #ignoring bias

    #embedding

    coords = dnx.chimera_coordinates(16, 16, 4)
    embedding = {}
    unit_num = (n + 3) // 4
    for i in range(0, n): 
        row = i + row_base * 4
        col = i + col_base * 4
        horizontal_unit = row // 4
        vertical_unit = col // 4
        embedding[str(i) + "_0"] = [coords.chimera_to_linear((j, vertical_unit, 0, i % 4)) for j in range(row_base, row_base + unit_num)]
        embedding[str(i) + "_1"] = [coords.chimera_to_linear((horizontal_unit, j, 1, i % 4)) for j in range(col_base, col_base + unit_num)]

    # new weights

    h_ = {}
    J_ = {}
    for i in range(0, n): 
        assert chain_strength >= 0
        J_[(str(i) + "_0", str(i) + "_1")] = - chain_strength

        for j in range(i + 1, n): 
            J_[(str(i) + "_0", str(j) + "_1")] = J[(i, j)] / 2
            J_[(str(j) + "_0", str(i) + "_1")] = J[(i, j)] / 2

    return (embedding, h_, J_)

def get_clique_embedding(n: int, h: dict, J: dict, chain_strength, row_base = 0, col_base = 0): #ignoring bias

    #embedding

    coords = dnx.chimera_coordinates(16, 16, 4)
    embedding = {}
    unit_num = (n + 3) // 4
    for i in range(0, n): 
        row = i + row_base * 4
        col = i + col_base * 4
        horizontal_unit = row // 4
        vertical_unit = col // 4
        embedding[str(i) + "_0"] = [coords.chimera_to_linear((j, vertical_unit, 0, i % 4)) for j in range(row_base, row_base + unit_num)]
        embedding[str(i) + "_1"] = [coords.chimera_to_linear((horizontal_unit, j, 1, i % 4)) for j in range(col_base, col_base + unit_num)]

    # new weights

    h_ = {}
    J_ = {}
    for i in range(0, n): 
        assert chain_strength >= 0
        J_[(str(i) + "_0", str(i) + "_1")] = - chain_strength

        for j in range(i + 1, n): 
            J_[(str(i) + "_0", str(j) + "_1")] = J[(i, j)] / 2
            J_[(str(j) + "_0", str(i) + "_1")] = J[(i, j)] / 2

    return (embedding, h_, J_)

def solve_exact(h, J): 
    cpu_sampleset = ExactSolver().sample_ising(h = h, J = J)
    print("CPU result: ")
    print(cpu_sampleset.slice(CUTOFF))
    return cpu_sampleset.first.energy

def solve_convert_cpu(h, J, n, chain_strength): 
    offset = n * chain_strength

    embedding, h_, J_ = get_clique_embedding_cpu(n, h, J, chain_strength, row_base=0, col_base=0)
    
    converted_model = BinaryQuadraticModel(h_, J_, offset, SPIN)
    sampler = SimulatedAnnealingSampler()
    print("CPU Embedding: ", embedding, J_)

    cpu_sampleset = sampler.sample(converted_model, num_reads = 1000)
    
    print("CPU result: ")
    print(cpu_sampleset.slice(CUTOFF))

    return cpu_sampleset.first.energy

def solve_convert_qpu(h, J, n, chain_strength): 
    offset = n * chain_strength

    embedding = None
    h_ = None
    J_ = None
    converted_model = None
    qpu_sampleset = None

    # for row_base in range(0, 16): 
    #     for col_base in range(0, 16): 
    for row_base in range(16): 
        for col_base in range(7, 16): 
            embedding, h_, J_ = get_clique_embedding(n, h, J, chain_strength, row_base=row_base, col_base=col_base)
            print(embedding)
            converted_model = BinaryQuadraticModel(h_, J_, offset, SPIN)
            try: 
                sampler = FixedEmbeddingComposite(DWaveSampler(solver='DW_2000Q_6'), embedding=embedding)
                qpu_sampleset = sampler.sample(converted_model, chain_strength = chain_strength, num_reads = 100, label = "custom_embedding_qpu_sampleset")

                print("embedding found. problem solved.")
                break
            except: 
                print("embedding failed. retrying")

        if qpu_sampleset is not None: 
            break

    assert qpu_sampleset, "No embedding found"
    print(embedding)
    
    print("QPU result: ")
    print(qpu_sampleset.slice(CUTOFF))

    return qpu_sampleset.first.energy, converted_model

if __name__ == "__main__":
    test_num = int(input())

    results = {}

    results["RATIO"] = RATIO
    results["COMPARE"] = RATIO
    results["exact_result"] = None
    results["convert_cpu_result"] = None
    results["my_result"] = None
    results["chain_strength"] = None

    with open("results.csv", "a") as f: 
        writer = csv.writer(f)
        writer.writerow([key for key in results.keys()])

        for test in range(test_num): 
            n = int(input())
            h, J = read_input(n)

            exact_result = solve_exact(h, J)

            original_model = BinaryQuadraticModel(h, J, 0, SPIN)

            chain_strength = get_my_chain_strength(n, h, J)
            convert_cpu_result = solve_convert_cpu(h, J, n, chain_strength)
            my_result, _ = solve_convert_qpu(h, J, n, chain_strength)

            results["exact_result"] = exact_result
            results["convert_cpu_result"] = convert_cpu_result
            results["my_result"] = my_result
            results["chain_strength"] = chain_strength
            results["COMPARE"] = "AC" if abs(exact_result - min(convert_cpu_result, my_result)) <= 1e-6 else "WA"

            writer.writerow([value for value in results.values()])
            print("-----------------------------------------------------------------------------")