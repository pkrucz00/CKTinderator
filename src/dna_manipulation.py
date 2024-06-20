def load_dna(path: str) -> str:
    with open(path, 'r') as file:
        return file.read().rstrip('\n')


def modify_gene_value(line: str, value: str):
    DOM_GENE_VAL_POS = 2
    splitted_line = line.split()
    splitted_line[DOM_GENE_VAL_POS] = value
    result =  " ".join(splitted_line)
    return result
 
def change_line_if_needed(line: str, genes_to_change: dict[str, int]) -> str:
    new_line = line
    for gene_name, gene_value in genes_to_change.items():
        if gene_name in line:
            new_line = modify_gene_value(line, str(gene_value))
    
    return new_line

def change_dna(dna_text: str, genes_to_change: dict[str, int]) -> str:    
    dna_lines = dna_text.split("\n")
    new_dna_lines = [change_line_if_needed(line, genes_to_change) for line in dna_lines]
    return "\n".join(new_dna_lines)

